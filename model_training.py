"""
model_training.py — Train and persist flight-delay prediction models.

Run directly:
    python model_training.py

Outputs:
    flight_delay_model.pkl   (Joblib bundle with models + encoders + scaler)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

from utils import engineer_features

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "flight_delay_model.pkl")

# ── 1. Load & Clean ──────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[1/5] Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"      Removed {before - len(df)} duplicates.")

    # Handle missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    # Sanity guards
    df = df[df["Distance"] >= 0]
    df = df[df["ScheduledDeparture"].between(0, 2359)]

    print(f"      Clean shape: {df.shape}")
    return df


# ── 2. Feature Engineering & Encoding ───────────────────────────────────────

def build_features(df: pd.DataFrame):
    df = engineer_features(df)

    cat_cols = ["Airline", "Origin", "Destination", "WeatherCondition"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "Airline", "Origin", "Destination",
        "ScheduledDeparture", "Distance", "DayOfWeek",
        "WeatherCondition", "DepartureHour",
        "IsRushHour", "IsLateNight", "IsWeekend",
        "IsLongHaul", "WeatherRisk",
    ]

    X = df[feature_cols].values
    y = df["Delayed"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"[2/5] Features: {len(feature_cols)}  |  Class balance: "
          f"{int(y.sum())} delayed / {int((1-y).sum())} on-time")

    return X, y, encoders, scaler, feature_cols


# ── 3. Train ─────────────────────────────────────────────────────────────────

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    print("\n[3/5] Training models …")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

        results[name] = {
            "model":      model,
            "accuracy":   accuracy_score(y_test, y_pred),
            "auc":        roc_auc_score(y_test, y_prob),
            "cv_mean":    cv_scores.mean(),
            "cv_std":     cv_scores.std(),
            "report":     classification_report(y_test, y_pred, output_dict=True),
            "confusion":  confusion_matrix(y_test, y_pred),
        }
        print(f"      {name:25s}  acc={results[name]['accuracy']:.3f}  "
              f"auc={results[name]['auc']:.3f}  "
              f"cv={results[name]['cv_mean']:.3f}±{results[name]['cv_std']:.3f}")

    # Best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n      ★ Best model: {best_name}")
    return results, best_name, X_test, y_test


# ── 4. Save ──────────────────────────────────────────────────────────────────

def save_bundle(results, best_name, encoders, scaler, feature_cols):
    bundle = {
        "models":       {n: r["model"] for n, r in results.items()},
        "best_model":   best_name,
        "encoders":     encoders,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "metrics":      {n: {k: v for k, v in r.items() if k != "model"}
                         for n, r in results.items()},
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n[4/5] Model bundle saved → {MODEL_PATH}")
    return bundle


# ── 5. Quick diagnostic plots (saved to disk, not shown interactively) ───────

def save_diagnostic_plots(results, feature_cols):
    out_dir = BASE_DIR
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(best["confusion"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["On-Time", "Delayed"],
                yticklabels=["On-Time", "Delayed"], ax=ax)
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # Feature importance (Random Forest)
    if "Random Forest" in results:
        rf = results["Random Forest"]["model"]
        imp = rf.feature_importances_
        idx = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(imp)), imp[idx], color="#3b82f6")
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels([feature_cols[i] for i in idx], rotation=45, ha="right")
        ax.set_title("Feature Importance — Random Forest")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=120)
        plt.close(fig)

    print(f"[5/5] Diagnostic plots saved.")


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    df                              = load_and_clean(DATA_PATH)
    X, y, encoders, scaler, f_cols  = build_features(df)
    results, best_name, Xt, yt      = train_models(X, y)
    save_bundle(results, best_name, encoders, scaler, f_cols)
    save_diagnostic_plots(results, f_cols)
    print("\n✅  Training complete.\n")
    return results


if __name__ == "__main__":
    run()
