import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import csv
import math
from scipy.special import ndtr
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import time

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

EPOCHE = 50

def gelu_exact_torch(x):
    return F.gelu(x)


def gelu_prime_exact_torch(x):
    phi = torch.tensor(ndtr(x.detach().numpy()), dtype=x.dtype)
    pdf = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    return phi + x * pdf


def chebyshev_coeffs(func, a, b, n):
    n = n + 1
    bma = 0.5 * (b - a)
    bpa = 0.5 * (b + a)
    f = [func(math.cos(math.pi * (k + 0.5) / n) * bma + bpa) for k in range(n)]
    fac = 2.0 / n
    return [
        fac * sum(f[k] * math.cos(math.pi * j * (k + 0.5) / n) for k in range(n))
        for j in range(n)
    ]


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

_coeffs_fwd = torch.tensor(
    chebyshev_coeffs(gelu_numpy, GELU_A, GELU_B, GELU_DEGREE), dtype=torch.float64
)
_coeffs_der = torch.tensor(
    chebyshev_coeffs(gelu_prime_numpy, GELU_A, GELU_B, GELU_DEGREE), dtype=torch.float64
)


def gelu_poly_torch(x):
    return _clenshaw_eval(x, _coeffs_fwd.to(x.device), GELU_A, GELU_B)


def gelu_prime_poly_torch(x):
    return _clenshaw_eval(x, _coeffs_der.to(x.device), GELU_A, GELU_B)


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


class Rete(nn.Module):
    def __init__(self, W1_np, b1_np, W2_np, b2_np):
        super().__init__()
        self.W1 = nn.Parameter(torch.tensor(W1_np, dtype=torch.float64))
        self.b1 = nn.Parameter(torch.tensor(b1_np, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.tensor(W2_np, dtype=torch.float64))
        self.b2 = nn.Parameter(torch.tensor(b2_np, dtype=torch.float64))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = gelu_train(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2

    def predict_exact(self, X):
        with torch.no_grad():
            z1 = X @ self.W1 + self.b1
            a1 = gelu_exact_torch(z1)
            z2 = a1 @ self.W2 + self.b2
        return torch.argmax(z2, dim=1)


def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = "esatta" if USE_EXACT_GELU else "approssimata"
    csv_file = os.path.join(script_dir, f"iris_pytorch_results_{suffix}.csv")

    print(f"  CSV output: {csv_file}")

    results = []

    iris = load_iris()
    X, y = iris.data, iris.target
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    def one_hot(e, n):
        oh = np.zeros((len(e), n))
        oh[np.arange(len(e)), e] = 1.0
        return oh

    y_tr_oh = one_hot(y_tr, 3)

    np.random.seed(42)
    W1 = np.random.randn(4, 8) * np.sqrt(1.0 / 4)
    W2 = np.random.randn(8, 3) * np.sqrt(1.0 / 8)
    b1 = np.zeros(8)
    b2 = np.zeros(3)

    net = Rete(W1, b1, W2, b2)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float64)
    X_te_t = torch.tensor(X_te, dtype=torch.float64)
    y_tr_t = torch.tensor(y_tr_oh, dtype=torch.float64)

    lr = 0.1

    N = X_tr_t.shape[0]
    tempo_train = 0.0
    for epoca in range(EPOCHE):
        t0 = time.perf_counter()
        with torch.no_grad():

            z1 = X_tr_t @ net.W1 + net.b1
            a1 = gelu_train(z1)
            z2 = a1 @ net.W2 + net.b2

            dz2      = (2.0 / N) * (z2 - y_tr_t)
            dW2      = a1.T @ dz2
            db2      = dz2.sum(dim=0)
            der_out1 = dz2 @ net.W2.T
            dgelu    = gelu_train_prime(z1)
            dz1      = der_out1 * dgelu
            dW1      = X_tr_t.T @ dz1
            db1      = dz1.sum(dim=0)

            net.W1 -= lr * dW1
            net.b1 -= lr * db1
            net.W2 -= lr * dW2
            net.b2 -= lr * db2

        tempo_train += time.perf_counter() - t0

        y_pred_tr = net.predict_exact(X_tr_t).numpy()
        y_pred_te = net.predict_exact(X_te_t).numpy()

        tr_acc  = np.mean(y_pred_tr == y_tr)
        te_acc  = np.mean(y_pred_te == y_te)
        f1_tr   = f1_score(y_tr, y_pred_tr, average='macro', zero_division=0)
        f1_te   = f1_score(y_te, y_pred_te, average='macro', zero_division=0)
        prec_tr = precision_score(y_tr, y_pred_tr, average='macro', zero_division=0)
        prec_te = precision_score(y_te, y_pred_te, average='macro', zero_division=0)
        rec_tr  = recall_score(y_tr, y_pred_tr, average='macro', zero_division=0)
        rec_te  = recall_score(y_te, y_pred_te, average='macro', zero_division=0)
        cm_te   = confusion_matrix(y_te, y_pred_te)
        tp_str, fp_str, fn_str, tn_str = cm_to_tp_fp_fn_tn(cm_te)

        results.append({
            "epoca": epoca + 1,
            "acc_train": tr_acc,
            "acc_test": te_acc,
            "f1_train": f1_tr,
            "f1_test": f1_te,
            "prec_train": prec_tr,
            "prec_test": prec_te,
            "rec_train": rec_tr,
            "rec_test": rec_te,
            "mse_train": calcola_mse_pytorch(net, X_tr, one_hot(y_tr, 3)),
            "mse_test": calcola_mse_pytorch(net, X_te, one_hot(y_te, 3)),

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

        print(f"  Epoch {epoca+1:3d} | "
              f"acc={tr_acc:.4f}/{te_acc:.4f} | "
              f"f1={f1_tr:.4f}/{f1_te:.4f} | "
              f"prec={prec_tr:.4f}/{prec_te:.4f} | "
              f"rec={rec_tr:.4f}/{rec_te:.4f}")
        if (epoca + 1) % 10 == 0:
            print(f"    CM test: {cm_te[0]}")
            print(f"             {cm_te[1]}")
            print(f"             {cm_te[2]}")

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRisultati salvati in: {csv_file}")
    print(f"  Tempo di training ({EPOCHE} epoche): {tempo_train:.4f} s")

    print("=" * 58)
    y_pred_te_fin = net.predict_exact(X_te_t).numpy()
    f1_fin   = f1_score(y_te, y_pred_te_fin, average='macro', zero_division=0)
    prec_fin = precision_score(y_te, y_pred_te_fin, average='macro', zero_division=0)
    rec_fin  = recall_score(y_te, y_pred_te_fin, average='macro', zero_division=0)
    cm_fin   = confusion_matrix(y_te, y_pred_te_fin)
    print(f"  Accuracy finale test: {np.mean(y_pred_te_fin == y_te):.4f}")
    print(f"  F1 macro:   {f1_fin:.4f}")
    print(f"  Precision:  {prec_fin:.4f}")
    print(f"  Recall:     {rec_fin:.4f}")
    print(f"  CM test: {cm_fin[0]}")
    print(f"           {cm_fin[1]}")
    print(f"           {cm_fin[2]}")
    print("=" * 58)


if __name__ == "__main__":
    train()