import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================
# CONFIG — cambia solo DATASET
# =============================

DATASET = "wine"  # "iris" | "wine" | "cancer" | "digits"

CONFIG = {
    "iris": {
        "prefix": "iris",
        "ckks": [
            "iris_ckks_results_sicuro.csv",
            "iris_ckks_results_non_sicuro.csv",
        ],
        "out": "plots_iris",
    },
    "wine": {
        "prefix": "wine",
        "ckks": [
            "wine_ckks_results_sicuro.csv",
            "wine_ckks_results_non_sicuro.csv",
        ],
        "out": "plots_wine",
    },
    "cancer": {
        "prefix": "cancer",
        "ckks": [
            "cancer_ckks_results_sicuro.csv",
            "cancer_ckks_results_non_sicuro.csv",
        ],
        "out": "plots_cancer",
    },
    "digits": {
        "prefix": "digits",
        "ckks": [
            "digits_ckks_results_sicuro.csv",
            "digits_ckks_results_non_sicuro.csv",
        ],
        "out": "plots_digits",
    },
}

cfg = CONFIG[DATASET]
CKKS_CSV_CANDIDATES = cfg["ckks"]
PT_POLY_CSV = f"{cfg['prefix']}_pytorch_results_approssimata.csv"
PT_EXACT_CSV = f"{cfg['prefix']}_pytorch_results_esatta.csv"
OUT = cfg["out"]
os.makedirs(OUT, exist_ok=True)

# =============================
# DIMENSIONI E FONT
# =============================

FULL_W = 7.0  
HALF_W = 3.5  
FULL_H = 3.2
HALF_H = 2.4

Y_MIN, Y_MAX = 0.80, 1.03 

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11,
})


# =============================
# LOAD
# =============================


def find_ckks_csv():
  for name in CKKS_CSV_CANDIDATES:
    if os.path.exists(name):
      return name
  return None


ckks_path = find_ckks_csv()
ckks = pd.read_csv(ckks_path) if ckks_path else None
if ckks is None:
  print(f"  [attenzione] nessun CSV CKKS trovato tra {CKKS_CSV_CANDIDATES}")
else:
  print(f"CKKS: uso {ckks_path}")

pt_poly = pd.read_csv(PT_POLY_CSV) if os.path.exists(PT_POLY_CSV) else None
pt_exact = pd.read_csv(PT_EXACT_CSV) if os.path.exists(PT_EXACT_CSV) else None

if pt_poly is None:
  print(f"  [attenzione] {PT_POLY_CSV} non trovato")
if pt_exact is None:
  print(f"  [attenzione] {PT_EXACT_CSV} non trovato")

SOURCES = [
    (pt_exact, "GELU esatta", "tab:blue", "-", 2.5, 1),
    (pt_poly, "Chebyshev poly (chiaro)", "tab:orange", "--", 2.0, 2),
    (ckks, "CKKS (FHE)", "tab:green", ":", 1.8, 3),
]


def save_plot(name):
  plt.tight_layout()
  plt.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight")
  plt.close()


def add_headroom(frac):
  """Allunga l'asse y verso l'alto per far stare la legenda senza coprire le curve."""
  ymin, ymax = plt.ylim()
  plt.ylim(ymin, ymax + frac * (ymax - ymin))


# =============================
# 1) ACCURACY / F1 / MSE — train | test
# =============================


def train_test_subplots(
    col_train,
    col_test,
    ylabel,
    title,
    filename,
    logy=False,
    legend_loc="best",
):
  fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_H), sharey=not logy)
  any_data = False
  for ax, col, split_label in [
      (axes[0], col_train, "train"),
      (axes[1], col_test, "test"),
  ]:
    for df, label, color, ls, lw, z in SOURCES:
      if df is not None and len(df) > 0 and col in df.columns:
        ax.plot(
            df["epoca"],
            df[col],
            label=label,
            color=color,
            linestyle=ls,
            linewidth=lw,
            zorder=z,
            alpha=0.85,
        )
        any_data = True
    ax.set_xlabel("Epoca")
    ax.set_title(split_label, fontweight="bold")
    ax.grid(alpha=0.3)
    if logy:
      ax.set_yscale("log")

  if not any_data:
    plt.close(fig)
    return

  axes[0].set_ylabel(ylabel)
  axes[1].legend(loc=legend_loc, framealpha=0.9)
  fig.suptitle(title, y=1.02)
  save_plot(filename)


train_test_subplots(
    "acc_train",
    "acc_test",
    "Accuracy",
    "Accuracy — GELU esatta vs Chebyshev vs CKKS",
    "accuracy.png",
)
train_test_subplots(
    "f1_train",
    "f1_test",
    "F1 score",
    "F1 macro — GELU esatta vs Chebyshev vs CKKS",
    "f1.png",
)
train_test_subplots(
    "mse_train",
    "mse_test",
    "MSE (log)",
    "MSE — GELU esatta vs Chebyshev vs CKKS",
    "mse.png",
    logy=True,
    legend_loc="best",
)


# =============================
# 2) PRECISION & RECALL
# =============================

fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_H), sharey=True)
any_data = False
for ax, col, name in [
    (axes[0], "prec_test", "Precision (test)"),
    (axes[1], "rec_test", "Recall (test)"),
]:
  for df, label, color, ls, lw, z in SOURCES:
    if df is not None and len(df) > 0 and col in df.columns:
      ax.plot(
          df["epoca"],
          df[col],
          label=label,
          color=color,
          linestyle=ls,
          linewidth=lw,
          zorder=z,
          alpha=0.85,
      )
      any_data = True
  ax.set_xlabel("Epoca")
  ax.set_title(name)
  ax.grid(alpha=0.3)
if any_data:
  axes[0].set_ylabel("Score")
  axes[1].legend(loc="best")
  fig.suptitle("Precision & Recall (test) — GELU esatta vs Chebyshev vs CKKS")
  save_plot("precision_recall_test.png")
else:
  plt.close(fig)


# =============================
# 3) DECOMPOSIZIONE DELL'ERRORE — assoluta
# =============================

have_poly_exact = (
    pt_exact is not None
    and pt_poly is not None
    and len(pt_exact) > 0
    and len(pt_poly) > 0
    and "mse_test" in pt_exact.columns
    and "mse_test" in pt_poly.columns
)

if have_poly_exact:
  n = min(len(pt_exact), len(pt_poly))
  epoche = pt_exact["epoca"].iloc[:n]
  err_poly = np.abs(
      pt_poly["mse_test"].iloc[:n].values - pt_exact["mse_test"].iloc[:n].values
  )

  plt.figure(figsize=(HALF_W, HALF_H))
  plt.plot(
      epoche,
      err_poly,
      label="Approssimazione polinomiale",
      color="tab:orange",
      linestyle="--",
      linewidth=2.0,
  )

  if ckks is not None and len(ckks) > 0 and "mse_test" in ckks.columns:
    n2 = min(len(ckks), len(pt_poly))
    err_fhe = np.abs(
        ckks["mse_test"].iloc[:n2].values - pt_poly["mse_test"].iloc[:n2].values
    )
    plt.plot(
        ckks["epoca"].iloc[:n2],
        err_fhe,
        label="Rumore CKKS",
        color="tab:green",
        linestyle=":",
        linewidth=2.0,
    )

  plt.xlabel("Epoca")
  plt.ylabel("Differenza assoluta di MSE")
  plt.grid(alpha=0.3)
  add_headroom(0.35)
  save_plot("error_decomposition.png")
else:
  print("  [skip] error_decomposition.png: mancano mse_test")


# =============================
# 3b) DECOMPOSIZIONE DELL'ERRORE — percentuale
# =============================

if have_poly_exact:
  n = min(len(pt_exact), len(pt_poly))
  epoche = pt_exact["epoca"].iloc[:n]
  mse_esatta = pt_exact["mse_test"].iloc[:n].values
  err_poly_pct = (
      np.abs(pt_poly["mse_test"].iloc[:n].values - mse_esatta)
      / mse_esatta
      * 100
  )

  plt.figure(figsize=(HALF_W, HALF_H))
  plt.plot(
      epoche,
      err_poly_pct,
      label="Approssimazione polinomiale",
      color="tab:orange",
      linestyle="--",
      linewidth=2.0,
  )

  if ckks is not None and len(ckks) > 0 and "mse_test" in ckks.columns:
    n2 = min(len(ckks), len(pt_poly), len(pt_exact))
    mse_esatta2 = pt_exact["mse_test"].iloc[:n2].values
    err_fhe_pct = (
        np.abs(
            ckks["mse_test"].iloc[:n2].values
            - pt_poly["mse_test"].iloc[:n2].values
        )
        / mse_esatta2
        * 100
    )
    plt.plot(
        ckks["epoca"].iloc[:n2],
        err_fhe_pct,
        label="Rumore CKKS",
        color="tab:green",
        linestyle=":",
        linewidth=2.0,
    )

  plt.xlabel("Epoca")
  plt.ylabel("Scostamento relativo (%)")
  plt.grid(alpha=0.3)
  add_headroom(0.35)
  plt.legend(loc="best")
  save_plot("error_decomposition_percent.png")
else:
  print("  [skip] error_decomposition_percent.png: mancano mse_test")


# =============================
# 4) NORME DEI PESI
# =============================

for m, ylab in [
    ("norm_W1", r"$\Vert W_1 \Vert$"),
    ("norm_W2", r"$\Vert W_2 \Vert$"),
]:
  plt.figure(figsize=(HALF_W, HALF_H))
  plotted = False
  for df, label, color, ls, lw, z in SOURCES:
    if df is not None and len(df) > 0 and m in df.columns:
      plt.plot(
          df["epoca"],
          df[m],
          label=label,
          color=color,
          linestyle=ls,
          linewidth=lw,
          zorder=z,
          alpha=0.85,
      )
      plotted = True
  if plotted:
    plt.xlabel("Epoca")
    plt.ylabel(ylab)
    plt.title(f"Norma L2 dei pesi — {ylab}")
    plt.grid(alpha=0.3)
    add_headroom(0.5)
    if m == "norm_W2":
      plt.legend(loc="best")
    save_plot(f"{m}.png")
  else:
    plt.close()


# =============================
# 5) TEMPO DI COMPUTAZIONE
# =============================

if (
    ckks is not None
    and len(ckks) > 0
    and "tempo_cumulativo_s" in ckks.columns
):
  plt.figure(figsize=(HALF_W, HALF_H))
  plt.plot(
      ckks["epoca"],
      ckks["tempo_cumulativo_s"] / 3600,
      color="tab:green",
      linewidth=2.0,
  )
  plt.xlabel("Epoca")
  plt.ylabel("Tempo cumulativo (ore)")
  plt.title("Costo cumulativo CKKS")
  plt.grid(alpha=0.3)
  save_plot("ckks_cumulative_time.png")

if ckks is not None and len(ckks) > 0 and "tempo_s" in ckks.columns:
  plt.figure(figsize=(HALF_W, HALF_H))
  plt.plot(ckks["epoca"], ckks["tempo_s"], color="tab:green", linewidth=2.0)
  plt.xlabel("Epoca")
  plt.ylabel("Tempo (s)")
  plt.title("Tempo per epoca — CKKS")
  plt.grid(alpha=0.3)
  save_plot("ckks_time_per_epoca.png")


# =============================
# 6) LIVELLO CIPHERTEXT
# =============================

if (
    ckks is not None
    and len(ckks) > 0
    and "livello_W1_post_update" in ckks.columns
):
  plt.figure(figsize=(HALF_W, HALF_H))
  plt.plot(
      ckks["epoca"],
      ckks["livello_W1_post_update"],
      color="tab:green",
      linewidth=2.0,
  )
  plt.xlabel("Epoca")
  plt.ylabel("Livello ciphertext")
  plt.title("Livelli CKKS prima del bootstrap (W1)")
  plt.grid(alpha=0.3)
  save_plot("ckks_ciphertext_level.png")


# =============================
# 7) BAR CHART FINALE
# =============================

METRIC_LABELS = {
    "acc_test": "Accuracy",
    "f1_test": "F1 macro",
    "prec_test": "Precision macro",
    "rec_test": "Recall macro",
}

metrics = ["acc_test", "f1_test", "prec_test", "rec_test"]
metrics_avail = [
    m for m in metrics if pt_exact is not None and m in pt_exact.columns
]

if metrics_avail:
  fig, ax = plt.subplots(figsize=(FULL_W, 3.6))
  x = np.arange(len(metrics_avail))
  width = 0.25
  for i, (df, label, color, ls, lw, z) in enumerate(SOURCES):
    offset = (i - 1) * width
    if (
        df is not None
        and len(df) > 0
        and all(m in df.columns for m in metrics_avail)
    ):
      vals = [df[m].iloc[-1] for m in metrics_avail]
      bars = ax.bar(
          x + offset, vals, width, label=label, color=color, alpha=0.85
      )
      for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.002,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
  ax.set_xticks(x)
  ax.set_xticklabels([METRIC_LABELS[m] for m in metrics_avail])
  ax.set_ylim(Y_MIN, Y_MAX)
  ax.set_yticks(np.arange(Y_MIN, 1.001, 0.04))
  ax.set_ylabel("Score (asse troncato)")
  ax.set_title("Confronto finale (test set, ultima epoca)")
  ax.grid(axis="y", alpha=0.3)
  ax.legend(
      loc="upper center",
      bbox_to_anchor=(0.5, -0.08),
      ncol=3,
      frameon=False,
  )
  save_plot("final_metrics_comparison.png")


print(f"\nGrafici ({DATASET}) creati nella cartella:", OUT)