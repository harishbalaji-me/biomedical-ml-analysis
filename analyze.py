"""
Biomedical ML Analysis — Heart Disease Prediction
==================================================
Generates synthetic patient data, runs EDA, trains an ML model,
and saves charts + a summary report to the outputs/ folder.

Usage:
    python analyze.py                  # uses built-in synthetic data
    python analyze.py --csv your.csv   # uses your own CSV file
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works headless on any OS)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
OUTPUT_DIR   = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  DATA
# ──────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "rest_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "slope", "num_vessels", "thal"
]
TARGET = "heart_disease"


def generate_synthetic_data(n: int = 600, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return a synthetic but medically plausible heart-disease dataset."""
    rng = np.random.default_rng(seed)

    age       = rng.integers(29, 77, n)
    sex       = rng.integers(0, 2, n)
    cp        = rng.integers(0, 4, n)
    trestbps  = rng.integers(94, 200, n)
    chol      = rng.integers(126, 564, n)
    fbs       = (rng.random(n) < 0.15).astype(int)
    restecg   = rng.integers(0, 3, n)
    thalach   = rng.integers(71, 202, n)
    exang     = (rng.random(n) < 0.33).astype(int)
    oldpeak   = np.round(rng.uniform(0, 6.2, n), 1)
    slope     = rng.integers(0, 3, n)
    ca        = rng.integers(0, 4, n)
    thal      = rng.choice([1, 2, 3], n)

    # Logistic score → binary label (mimics Cleveland dataset statistics)
    score = (
        0.04 * age
        - 0.5 * sex
        + 0.3 * cp
        + 0.01 * trestbps
        + 0.003 * chol
        + 0.2 * fbs
        - 0.01 * thalach
        + 0.5 * exang
        + 0.3 * oldpeak
        + 0.2 * ca
        + 0.15 * thal
        - 5.0
    )
    prob    = 1 / (1 + np.exp(-score))
    disease = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "age": age, "sex": sex, "chest_pain_type": cp,
        "resting_bp": trestbps, "cholesterol": chol,
        "fasting_blood_sugar": fbs, "rest_ecg": restecg,
        "max_heart_rate": thalach, "exercise_angina": exang,
        "st_depression": oldpeak, "slope": slope,
        "num_vessels": ca, "thal": thal,
        TARGET: disease
    })


def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        print(f"[INFO] Loading user CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        # Accept Cleveland-style column names automatically
        rename = {
            "target": TARGET, "num": TARGET, "condition": TARGET,
            "trestbps": "resting_bp", "chol": "cholesterol",
            "fbs": "fasting_blood_sugar", "restecg": "rest_ecg",
            "thalach": "max_heart_rate", "exang": "exercise_angina",
            "oldpeak": "st_depression", "ca": "num_vessels",
        }
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
        df[TARGET] = (df[TARGET] > 0).astype(int)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
    else:
        print("[INFO] No CSV provided — using built-in synthetic data (600 patients).")
        df = generate_synthetic_data()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS  →  outputs/01_eda.png
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {"No Disease": "#4FC3F7", "Disease": "#EF5350"}


def plot_eda(df: pd.DataFrame) -> None:
    print("[INFO] Generating EDA chart …")
    df_plot = df.copy()
    df_plot["Status"] = df_plot[TARGET].map({0: "No Disease", 1: "Disease"})

    fig = plt.figure(figsize=(18, 12), facecolor="#0D1117")
    fig.suptitle(
        "Heart Disease — Exploratory Data Analysis",
        fontsize=22, fontweight="bold", color="white", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    plot_cfg = [
        (gs[0, 0], "age",           "Age Distribution by Status",       True),
        (gs[0, 1], "max_heart_rate","Max Heart Rate by Status",         True),
        (gs[0, 2], "cholesterol",   "Cholesterol by Status",            True),
        (gs[1, 0], "resting_bp",    "Resting Blood Pressure by Status", True),
        (gs[1, 1], "st_depression", "ST Depression by Status",          True),
    ]

    for spec, col, title, _ in plot_cfg:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        for status, color in PALETTE.items():
            subset = df_plot[df_plot["Status"] == status][col]
            ax.hist(subset, bins=20, alpha=0.7, label=status, color=color, edgecolor="none")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="#8B949E")
        ax.set_xlabel(col.replace("_", " ").title(), color="#8B949E", fontsize=9)
        ax.legend(fontsize=8, facecolor="#21262D", labelcolor="white")

    # Disease prevalence bar
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#161B22")
    for spine in ax6.spines.values():
        spine.set_color("#30363D")
    counts = df_plot["Status"].value_counts()
    bars = ax6.bar(counts.index, counts.values,
                   color=[PALETTE[k] for k in counts.index], edgecolor="none", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
    ax6.set_title("Patient Count by Status", color="white", fontsize=11, pad=8)
    ax6.tick_params(colors="#8B949E")
    ax6.set_ylabel("Patients", color="#8B949E", fontsize=9)

    path = os.path.join(OUTPUT_DIR, "01_eda.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CORRELATION HEATMAP  →  outputs/02_correlation.png
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame) -> None:
    print("[INFO] Generating correlation heatmap …")
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    corr = df[FEATURES + [TARGET]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        annot=True, fmt=".2f", linewidths=0.5, linecolor="#30363D",
        annot_kws={"size": 7, "color": "white"},
        ax=ax, cbar_kws={"shrink": 0.7}
    )
    ax.set_title("Feature Correlation Matrix", color="white", fontsize=16, pad=14)
    ax.tick_params(colors="#8B949E", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.patch.set_facecolor("#0D1117")

    path = os.path.join(OUTPUT_DIR, "02_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MODEL TRAINING + EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> dict:
    print("[INFO] Training ML models …")
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]),
    }

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        y_pred    = pipe.predict(X_test)
        y_prob    = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "pipe":      pipe,
            "cv_auc":    cv_scores,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "fpr":       fpr,
            "tpr":       tpr,
            "roc_auc":   auc(fpr, tpr),
            "report":    classification_report(y_test, y_pred, output_dict=True),
            "cm":        confusion_matrix(y_test, y_pred),
        }
        print(
            f"  {name:25s}  CV AUC={cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
            f"  Test AUC={results[name]['roc_auc']:.3f}"
        )

    results["_split"] = (X_train, X_test, y_train, y_test)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MODEL CHARTS  →  outputs/03_roc_curves.png  +  outputs/04_confusion.png
# ──────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = {"Logistic Regression": "#4FC3F7",
                "Random Forest":       "#81C784",
                "Gradient Boosting":   "#FFB74D"}


def plot_roc(results: dict) -> None:
    print("[INFO] Generating ROC curves …")
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    for spine in ax.spines.values():
        spine.set_color("#30363D")

    for name, color in MODEL_COLORS.items():
        r = results[name]
        ax.plot(r["fpr"], r["tpr"], color=color, lw=2.5,
                label=f"{name}  (AUC = {r['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], ":", color="#8B949E", lw=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", color="#8B949E")
    ax.set_ylabel("True Positive Rate",  color="#8B949E")
    ax.set_title("ROC Curves — Model Comparison", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#8B949E")
    ax.legend(facecolor="#21262D", labelcolor="white", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    path = os.path.join(OUTPUT_DIR, "03_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {path}")


def plot_confusion(results: dict) -> None:
    print("[INFO] Generating confusion matrices …")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0D1117")
    fig.suptitle("Confusion Matrices", color="white", fontsize=15, y=1.02)

    for ax, (name, color) in zip(axes, MODEL_COLORS.items()):
        ax.set_facecolor("#161B22")
        cm   = results[name]["cm"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["No Disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="#8B949E")
        ax.xaxis.label.set_color("#8B949E")
        ax.yaxis.label.set_color("#8B949E")
        for text in ax.texts:
            text.set_color("white")

    path = os.path.join(OUTPUT_DIR, "04_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  FEATURE IMPORTANCE  →  outputs/05_feature_importance.png
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(results: dict) -> None:
    print("[INFO] Generating feature importance chart …")
    rf_clf = results["Random Forest"]["pipe"].named_steps["clf"]
    importances = pd.Series(rf_clf.feature_importances_, index=FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    for spine in ax.spines.values():
        spine.set_color("#30363D")

    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(importances)))
    ax.barh(importances.index, importances.values, color=colors, edgecolor="none", height=0.65)
    ax.set_xlabel("Importance Score", color="#8B949E")
    ax.set_title("Feature Importance (Random Forest)", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#8B949E")

    path = os.path.join(OUTPUT_DIR, "05_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  TEXT SUMMARY REPORT  →  outputs/summary_report.txt
# ──────────────────────────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame, results: dict) -> None:
    print("[INFO] Writing summary report …")
    X_train, X_test, y_train, y_test = results["_split"]
    best_name = max(MODEL_COLORS, key=lambda n: results[n]["roc_auc"])

    lines = [
        "=" * 62,
        "  BIOMEDICAL ML ANALYSIS — HEART DISEASE PREDICTION",
        "  Summary Report",
        "=" * 62,
        "",
        "DATASET",
        f"  Total patients   : {len(df):,}",
        f"  Features         : {len(FEATURES)}",
        f"  Disease cases    : {df[TARGET].sum()} ({df[TARGET].mean()*100:.1f}%)",
        f"  Healthy cases    : {(df[TARGET]==0).sum()} ({(df[TARGET]==0).mean()*100:.1f}%)",
        f"  Train / Test split: {len(X_train)} / {len(X_test)}",
        "",
        "MODEL RESULTS (Test Set)",
        "-" * 40,
    ]

    for name in MODEL_COLORS:
        r   = results[name]
        rep = r["report"]
        lines += [
            f"\n  {name}",
            f"    CV AUC   : {r['cv_auc'].mean():.4f} ± {r['cv_auc'].std():.4f}",
            f"    Test AUC : {r['roc_auc']:.4f}",
            f"    Accuracy : {rep['accuracy']:.4f}",
            f"    Precision: {rep['weighted avg']['precision']:.4f}",
            f"    Recall   : {rep['weighted avg']['recall']:.4f}",
            f"    F1 Score : {rep['weighted avg']['f1-score']:.4f}",
        ]

    lines += [
        "",
        "=" * 62,
        f"  Best Model: {best_name}  (AUC = {results[best_name]['roc_auc']:.4f})",
        "=" * 62,
        "",
        "OUTPUT FILES",
        "  outputs/01_eda.png                — EDA distributions",
        "  outputs/02_correlation.png         — correlation heatmap",
        "  outputs/03_roc_curves.png          — ROC curve comparison",
        "  outputs/04_confusion.png           — confusion matrices",
        "  outputs/05_feature_importance.png  — feature importance",
        "  outputs/summary_report.txt         — this file",
    ]

    path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SAVED] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Biomedical ML Heart Disease Analysis")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to your own CSV file (optional)")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  Biomedical ML Analysis — Heart Disease")
    print("=" * 50 + "\n")

    df      = load_data(args.csv)
    plot_eda(df)
    plot_correlation(df)
    results = train_models(df)
    plot_roc(results)
    plot_confusion(results)
    plot_feature_importance(results)
    write_report(df, results)

    print("\n✅  All done!  Check the outputs/ folder.\n")


if __name__ == "__main__":
    main()
