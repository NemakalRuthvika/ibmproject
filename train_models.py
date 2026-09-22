"""
train_models.py
───────────────
Trains Logistic Regression, Decision Tree, and Random Forest pipelines on the
Breast Cancer Wisconsin dataset and saves them to models/ via Joblib.

Run once before launching the Streamlit app:
    python train_models.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

RANDOM_STATE = 42
DATA_PATH = "data.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & clean ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
if "Unnamed: 32" in df.columns:
    df.drop(columns=["Unnamed: 32"], inplace=True)

FEATURE_COLS = [c for c in df.columns if c not in ("id", "diagnosis")]
TARGET_COL = "diagnosis_encoded"
df[TARGET_COL] = df["diagnosis"].map({"M": 1, "B": 0})

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Define pipelines ────────────────────────────────────────────────────────
pipelines = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10_000, random_state=RANDOM_STATE)),
    ]),
    "decision_tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    ]),
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ]),
}

# ── Train, evaluate, save ────────────────────────────────────────────────────
results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_test":    y_test.tolist(),
        "y_prob":    y_prob.tolist(),
    }
    results[name] = metrics
    path = os.path.join(MODEL_DIR, f"{name}_pipeline.pkl")
    joblib.dump(pipe, path)
    print(f"  [{name}]  acc={metrics['accuracy']}  auc={metrics['roc_auc']}  -> {path}")

# Save metrics + feature list so the app never re-trains
joblib.dump(results,     os.path.join(MODEL_DIR, "metrics.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))
joblib.dump((y_test.tolist(), ), os.path.join(MODEL_DIR, "test_labels.pkl"))

# Save RF feature importances
rf_pipe = pipelines["random_forest"]
importances = pd.Series(
    rf_pipe.named_steps["clf"].feature_importances_,
    index=FEATURE_COLS,
).sort_values(ascending=False)
joblib.dump(importances, os.path.join(MODEL_DIR, "rf_importances.pkl"))

print("\nAll models trained and saved to models/")
print("\n=== Final metrics ===")
for name, m in results.items():
    print(f"  {name:25s}  acc={m['accuracy']}  prec={m['precision']}  "
          f"rec={m['recall']}  f1={m['f1']}  auc={m['roc_auc']}")
