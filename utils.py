"""
utils.py
────────
Shared helpers for the Streamlit app.
Handles data loading, caching, model loading, and metric helpers.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import roc_curve

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data.csv"
MODEL_DIR  = "models"

PALETTE = {"Benign": "#3B82F6", "Malignant": "#EF4444"}   # blue / red
SNS_PALETTE = ["#3B82F6", "#EF4444"]

# ── Readable model labels ─────────────────────────────────────────────────────
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree":       "Decision Tree",
    "random_forest":       "Random Forest",
}

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Unnamed: 32" in df.columns:
        df.drop(columns=["Unnamed: 32"], inplace=True)
    df["diagnosis_label"] = df["diagnosis"].map({"B": "Benign", "M": "Malignant"})
    return df

@st.cache_data(show_spinner=False)
def get_feature_cols() -> list:
    df = load_data()
    return [c for c in df.columns if c not in ("id", "diagnosis", "diagnosis_label")]

# ── Models ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(key: str):
    path = os.path.join(MODEL_DIR, f"{key}_pipeline.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_metrics() -> dict:
    path = os.path.join(MODEL_DIR, "metrics.pkl")
    if not os.path.exists(path):
        return {}
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_rf_importances() -> pd.Series:
    path = os.path.join(MODEL_DIR, "rf_importances.pkl")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    return joblib.load(path)

def models_trained() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, "metrics.pkl"))

# ── Matplotlib helpers ────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#E5E7EB",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":       "#F3F4F6",
        "grid.linewidth":    0.8,
        "font.size":         11,
    })

def make_fig(nrows=1, ncols=1, figsize=(9, 4)):
    apply_style()
    return plt.subplots(nrows, ncols, figsize=figsize)


def plot_confusion_matrix(cm, ax, title=""):
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot  = np.array([[f"{labels[i][j]}\n{cm[i][j]}" for j in range(2)] for i in range(2)])
    cmap   = sns.light_palette("#3B82F6", as_cmap=True)
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, ax=ax,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                linewidths=0.5, linecolor="#E5E7EB",
                annot_kws={"size": 11})
    ax.set_xlabel("Predicted", fontsize=10, labelpad=6)
    ax.set_ylabel("Actual",    fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


def plot_roc_curves(metrics: dict, ax):
    colors = {"logistic_regression": "#3B82F6",
               "decision_tree":       "#F59E0B",
               "random_forest":       "#10B981"}
    for key, label in MODEL_LABELS.items():
        if key not in metrics:
            continue
        m = metrics[key]
        fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
        ax.plot(fpr, tpr, label=f"{label} (AUC={m['roc_auc']:.3f})",
                color=colors[key], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate",  fontsize=10)
    ax.set_title("ROC Curves — Model Comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
