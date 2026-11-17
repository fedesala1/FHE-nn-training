import math
import os
import csv
import numpy as np
import torch
from scipy.special import ndtr
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

SEP = "=" * 60

print(SEP)
print("MODALITÀ")
print(SEP)
risposta_modalita = input(
    "Training con GELU ESATTA o con GELU APPROSSIMATA "
    "(Chebyshev, come nel file FHE)? [e/a]: "
).strip().lower()
USE_EXACT_GELU = risposta_modalita in ("e", "esatta", "exact")

if USE_EXACT_GELU:
    print("  → modalità ESATTA: nessuna approssimazione polinomiale")
else:
    print("  → modalità APPROSSIMATA: Chebyshev grado 11 su [-6,6], come nel file FHE")
print(SEP)


def chebyshev_coeffs(func, a, b, n):
    n = n + 1
    bma = 0.5 * (b - a)
    bpa = 0.5 * (b + a)
    f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
    fac = 2.0 / n
    return [fac * sum(f[k] * math.cos(math.pi * j * (k + 0.5) / n) for k in range(n)) for j in range(n)]


def _clenshaw_eval(x, coeffs, a, b):
    t = (2.0 * x - (a + b)) / (b - a)
    n = len(coeffs)
    b_kp2 = torch.zeros_like(t)
    b_kp1 = torch.zeros_like(t)
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2.0 * t * b_kp1 - b_kp2
        b_kp2, b_kp1 = b_kp1, b_k
    return coeffs[0] / 2.0 + t * b_kp1 - b_kp2


def gelu_numpy(x):
    return x * ndtr(x)


def gelu_prime_numpy(x):
    return ndtr(x) + x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


GELU_A, GELU_B, GELU_DEGREE = -6.0, 6.0, 11

poly_coeffs_fwd = torch.tensor(
    chebyshev_coeffs(gelu_numpy, GELU_A, GELU_B, GELU_DEGREE), dtype=torch.float64
)
poly_coeffs_der = torch.tensor(
    chebyshev_coeffs(gelu_prime_numpy, GELU_A, GELU_B, GELU_DEGREE), dtype=torch.float64
)


def gelu_poly_torch(x):
    return _clenshaw_eval(x, poly_coeffs_fwd.to(x.device), GELU_A, GELU_B)


def gelu_prime_poly_torch(x):
    return _clenshaw_eval(x, poly_coeffs_der.to(x.device), GELU_A, GELU_B)


def gelu_exact_torch(x):
    return x * torch.special.ndtr(x)


def gelu_prime_exact_torch(x):
    return torch.special.ndtr(x) + x * torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


if USE_EXACT_GELU:
    gelu_train = gelu_exact_torch
    gelu_train_prime = gelu_prime_exact_torch
else:
    gelu_train = gelu_poly_torch
    gelu_train_prime = gelu_prime_poly_torch


def calcola_mse_pytorch(net, X, y_onehot):
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float64)
        z1 = X_t @ net.W1 + net.b1
        a1 = gelu_exact_torch(z1)
        z2 = a1 @ net.W2 + net.b2
        return torch.mean(
            (z2 - torch.tensor(y_onehot, dtype=torch.float64)) ** 2
        ).item()


def cm_to_tp_fp_fn_tn(cm):
    n_classes = cm.shape[0]
    total = cm.sum()
    tp_list, fp_list, fn_list, tn_list = [], [], [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        tp_list.append(str(int(tp)))
        fp_list.append(str(int(fp)))
        fn_list.append(str(int(fn)))
        tn_list.append(str(int(tn)))
    return ";".join(tp_list), ";".join(fp_list), ";".join(fn_list), ";".join(tn_list)


M_IN, L_HID, C_OUT = 30, 16, 1

SLOT_SECURE = 32768

N = SLOT_SECURE // (M_IN * L_HID)


class Rete:
    def __init__(self, W1_np, b1_np, W2_np, b2_np):
        self.W1 = torch.tensor(W1_np, dtype=torch.float64)
        self.b1 = torch.tensor(b1_np, dtype=torch.float64)
        self.W2 = torch.tensor(W2_np, dtype=torch.float64)
        self.b2 = torch.tensor(b2_np, dtype=torch.float64)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = gelu_train(z1)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def predict_exact(self, X_np):
        X = torch.tensor(X_np, dtype=torch.float64)
        z1 = X @ self.W1 + self.b1
        a1 = torch.tensor(gelu_numpy(z1.numpy()), dtype=torch.float64)
        z2 = a1 @ self.W2 + self.b2
        return (z2[:, 0] >= 0.5).numpy().astype(int)


def train():
    np.random.seed(42)

    LR = 0.1
    EPOCHE = 40
    CLIP_MIN, CLIP_MAX = -5, 5

    suffisso = "esatta" if USE_EXACT_GELU else "approssimata"
    csv_file = f"cancer_pytorch_results_{suffisso}.csv"
    results = []

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X = StandardScaler().fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    y_tr_oh = y_tr.reshape(-1, 1).astype(float)
    y_te_oh = y_te.reshape(-1, 1).astype(float)

    N_MINI_BATCHES = len(X_tr) // N
    print(f"Train size: {len(X_tr)}  ->  N_MINI_BATCHES = {N_MINI_BATCHES}  "
          f"(N={N}, {len(X_tr) - N_MINI_BATCHES*N} campioni scartati)")

    W1 = np.random.randn(M_IN, L_HID) * np.sqrt(1.0 / M_IN)
    b1 = np.zeros(L_HID)
    W2 = np.random.randn(L_HID, C_OUT) * np.sqrt(1.0 / L_HID)
    b2 = np.zeros(C_OUT)

    net = Rete(W1, b1, W2, b2)

    X_batches, y_batches = [], []
    for i in range(N_MINI_BATCHES):
        xb = np.clip(X_tr[i*N:(i+1)*N], CLIP_MIN, CLIP_MAX)
        yb = y_tr_oh[i*N:(i+1)*N]
        X_batches.append(torch.tensor(xb, dtype=torch.float64))
        y_batches.append(torch.tensor(yb, dtype=torch.float64))

    pt_tr0 = accuracy_score(y_tr, net.predict_exact(X_tr))
    pt_te0 = accuracy_score(y_te, net.predict_exact(X_te))
    print(f"Init      PT {pt_tr0:.4f}/{pt_te0:.4f}")

    for epoca in range(EPOCHE):

        dW1_acc = torch.zeros_like(net.W1)
        db1_acc = torch.zeros_like(net.b1)
        dW2_acc = torch.zeros_like(net.W2)
        db2_acc = torch.zeros_like(net.b2)

        with torch.no_grad():
            for i in range(N_MINI_BATCHES):
                X_b = X_batches[i]
                y_b = y_batches[i]

                z1, a1, z2 = net.forward(X_b)

                dz2      = (2.0 / N) * (z2 - y_b)
                db2      = dz2.sum(dim=0)
                dW2      = a1.T @ dz2
                der_out1 = dz2 @ net.W2.T
                dgelu    = gelu_train_prime(z1)
                dz1      = der_out1 * dgelu
                db1      = dz1.sum(dim=0)
                dW1      = X_b.T @ dz1

                dW1_acc += dW1
                db1_acc += db1
                dW2_acc += dW2
                db2_acc += db2

            dW1_ct = dW1_acc / N_MINI_BATCHES
            db1_ct = db1_acc / N_MINI_BATCHES
            dW2_ct = dW2_acc / N_MINI_BATCHES
            db2_ct = db2_acc / N_MINI_BATCHES

            net.W1 -= LR * dW1_ct
            net.b1 -= LR * db1_ct
            net.W2 -= LR * dW2_ct
            net.b2 -= LR * db2_ct

        y_pred_tr = net.predict_exact(X_tr)
        y_pred_te = net.predict_exact(X_te)

        acc_tr  = accuracy_score(y_tr, y_pred_tr)
        acc_te  = accuracy_score(y_te, y_pred_te)
        f1_tr   = f1_score(y_tr, y_pred_tr, average='binary', zero_division=0)
        f1_te   = f1_score(y_te, y_pred_te, average='binary', zero_division=0)
        prec_tr = precision_score(y_tr, y_pred_tr, average='binary', zero_division=0)
        prec_te = precision_score(y_te, y_pred_te, average='binary', zero_division=0)
        rec_tr  = recall_score(y_tr, y_pred_tr, average='binary', zero_division=0)
        rec_te  = recall_score(y_te, y_pred_te, average='binary', zero_division=0)
        cm_te   = confusion_matrix(y_te, y_pred_te)
        tp_str, fp_str, fn_str, tn_str = cm_to_tp_fp_fn_tn(cm_te)

        results.append({
            "epoca": epoca + 1,
            "acc_train": acc_tr,
            "acc_test": acc_te,
            "f1_train": f1_tr,
            "f1_test": f1_te,
            "prec_train": prec_tr,
            "prec_test": prec_te,
            "rec_train": rec_tr,
            "rec_test": rec_te,
            "mse_train": calcola_mse_pytorch(net, X_tr, y_tr_oh),
            "mse_test": calcola_mse_pytorch(net, X_te, y_te_oh),
            "norm_W1": torch.norm(net.W1).item(),
            "norm_W2": torch.norm(net.W2).item(),
            "norm_b1": torch.norm(net.b1).item(),
            "norm_b2": torch.norm(net.b2).item(),
            "max_W1": torch.max(torch.abs(net.W1)).item(),
            "max_W2": torch.max(torch.abs(net.W2)).item(),
            "tp_test": tp_str,
            "fp_test": fp_str,
            "fn_test": fn_str,
            "tn_test": tn_str,
        })

        print(f"  Epoca {epoca+1:3d} | "
              f"acc={acc_tr:.4f}/{acc_te:.4f} | "
              f"f1={f1_tr:.4f}/{f1_te:.4f} | "
              f"prec={prec_tr:.4f}/{prec_te:.4f} | "
              f"rec={rec_tr:.4f}/{rec_te:.4f}")
        if (epoca + 1) % 10 == 0:
            print(f"    CM test: {cm_te[0]}")
            print(f"             {cm_te[1]}")

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRisultati salvati in: {csv_file}")

    print("=" * 58)
    y_pred_te_fin = net.predict_exact(X_te)
    print(f"  Accuracy finale test: {accuracy_score(y_te, y_pred_te_fin):.4f}")


if __name__ == "__main__":
    train()