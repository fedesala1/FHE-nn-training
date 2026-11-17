import math
import numpy as np
import torch
import os
import csv
from scipy.special import ndtr
from sklearn.datasets import load_wine
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
    f = [func(math.cos(math.pi*(k+0.5)/n)*bma + bpa) for k in range(n)]
    fac = 2.0/n
    return [fac * sum(f[k]*math.cos(math.pi*j*(k+0.5)/n) for k in range(n)) for j in range(n)]

def get_poly_coeffs(a, b, n, func):
    c = chebyshev_coeffs(func, a, b, n)
    n = len(c)
    T_prev = np.zeros(n); T_prev[0] = 1
    T_curr = np.zeros(n); T_curr[0] = -(a+b)/(b-a); T_curr[1] = 2/(b-a)
    coeffs = c[0]/2 * T_prev + c[1] * T_curr
    for k in range(2, n):
        T_next = np.zeros(n)
        scale = 2/(b-a); shift = -(a+b)/(b-a)
        for i in range(n-1):
            T_next[i]   += 2*shift*T_curr[i]
            T_next[i+1] += 2*scale*T_curr[i]
        T_next -= T_prev
        coeffs += c[k] * T_next
        T_prev, T_curr = T_curr, T_next
    return torch.tensor(coeffs, dtype=torch.float64)

def gelu_numpy(x):
    return x * ndtr(x)
 
 
def gelu_prime_numpy(x):
    return ndtr(x) + x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
 
 
def _clenshaw_eval(x, coeffs, a, b):
    t = (2.0 * x - (a + b)) / (b - a)
    n = len(coeffs)
    b_kp2 = torch.zeros_like(t)
    b_kp1 = torch.zeros_like(t)
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2.0 * t * b_kp1 - b_kp2
        b_kp2, b_kp1 = b_kp1, b_k
    return coeffs[0] / 2.0 + t * b_kp1 - b_kp2
 
 
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


M_IN, L_HID, C_OUT = 13, 8, 3

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
        return torch.argmax(z2, dim=1).numpy()


def train():
    np.random.seed(42)

    N      = 142
    LR     = 0.1
    EPOCHE = 40

    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    suffisso = "esatta" if USE_EXACT_GELU else "approssimata"
    csv_file = os.path.join(output_dir, f"wine_pytorch_results_{suffisso}.csv")

    results = []

    wine = load_wine()
    X, y = wine.data, wine.target
    X    = StandardScaler().fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    def one_hot(e, n):
        oh = np.zeros((len(e), n))
        oh[np.arange(len(e)), e] = 1.0
        return oh

    y_tr_oh = one_hot(y_tr, C_OUT)

    print(f"Train size: {len(X_tr)}  (N={N}, un unico batch, nessuna "
          f"suddivisione in mini-batch, come nel file FHE)")

    W1 = np.random.randn(M_IN, L_HID) * np.sqrt(1.0 / M_IN)
    b1 = np.zeros(L_HID)
    W2 = np.random.randn(L_HID, C_OUT) * np.sqrt(1.0 / L_HID)
    b2 = np.zeros(C_OUT)

    net = Rete(W1, b1, W2, b2)

    X_b = torch.tensor(X_tr[:N], dtype=torch.float64)
    y_b = torch.tensor(y_tr_oh[:N], dtype=torch.float64)

    pt_tr0 = accuracy_score(y_tr, net.predict_exact(X_tr))
    pt_te0 = accuracy_score(y_te, net.predict_exact(X_te))
    print(f"Init      PT {pt_tr0:.4f}/{pt_te0:.4f}")

    for epoca in range(EPOCHE):

        with torch.no_grad():
            z1, a1, z2 = net.forward(X_b)

            dz2      = (2.0 / N) * (z2 - y_b)
            db2      = dz2.sum(dim=0)
            dW2      = a1.T @ dz2
            der_out1 = dz2 @ net.W2.T
            dgelu    = gelu_train_prime(z1)
            dz1      = der_out1 * dgelu
            db1      = dz1.sum(dim=0)
            dW1      = X_b.T @ dz1

            net.W1 -= LR * dW1
            net.b1 -= LR * db1
            net.W2 -= LR * dW2
            net.b2 -= LR * db2

        y_pred_tr = net.predict_exact(X_tr)
        y_pred_te = net.predict_exact(X_te)

        acc_tr  = accuracy_score(y_tr, y_pred_tr)
        acc_te  = accuracy_score(y_te, y_pred_te)
        f1_tr   = f1_score(y_tr, y_pred_tr, average='macro', zero_division=0)
        f1_te   = f1_score(y_te, y_pred_te, average='macro', zero_division=0)
        prec_tr = precision_score(y_tr, y_pred_tr, average='macro', zero_division=0)
        prec_te = precision_score(y_te, y_pred_te, average='macro', zero_division=0)
        rec_tr  = recall_score(y_tr, y_pred_tr, average='macro', zero_division=0)
        rec_te  = recall_score(y_te, y_pred_te, average='macro', zero_division=0)
        cm_te   = confusion_matrix(y_te, y_pred_te)

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
            "mse_train": calcola_mse_pytorch(net, X_tr, one_hot(y_tr, C_OUT)),
            "mse_test": calcola_mse_pytorch(net, X_te, one_hot(y_te, C_OUT)),
            "norm_W1": torch.norm(net.W1).item(),
            "norm_W2": torch.norm(net.W2).item(),
            "norm_b1": torch.norm(net.b1).item(),
            "norm_b2": torch.norm(net.b2).item(),
            "max_W1": torch.max(torch.abs(net.W1)).item(),
            "max_W2": torch.max(torch.abs(net.W2)).item(),
            "tp_test": cm_te.flatten().tolist()
        })

        print(f"  Epoca {epoca+1:3d} | "
              f"acc={acc_tr:.4f}/{acc_te:.4f} | "
              f"f1(macro)={f1_tr:.4f}/{f1_te:.4f} | "
              f"prec(macro)={prec_tr:.4f}/{prec_te:.4f} | "
              f"rec(macro)={rec_tr:.4f}/{rec_te:.4f}")
        if (epoca + 1) % 10 == 0:
            print(f"    CM test: {cm_te[0]}")
            print(f"             {cm_te[1]}")
            print(f"             {cm_te[2]}")

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