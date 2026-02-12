# src/cross_validation.py

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)

from train_svm import load_dataset

K = 5
MODEL_DIR = "models"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true",
                        help="Enable nested GridSearchCV inside each fold")
    parser.add_argument("--inner-cv", type=int, default=3,
                        help="Inner CV folds for GridSearch (default=3)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = time.time()

    print("[INFO] Loading training data...")
    X, y_fruit, _ = load_dataset()

    X = np.asarray(X, dtype=np.float32)

    le = LabelEncoder()
    y = le.fit_transform(y_fruit)
    y = np.asarray(y, dtype=np.int64)

    print(f"[INFO] Samples: {len(X)}")
    print(f"[INFO] Classes: {list(le.classes_)}")
    print(f"[INFO] Nested GridSearch: {'ON' if args.grid else 'OFF'}\n")

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    # 🔥 Expanded hyperparameter grid
    param_grid = {
        "svc__C": [1, 10, 50, 100],
        "svc__gamma": ["scale", "auto", 0.01, 0.001],
    }

    rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True))
        ])

        if args.grid:
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=args.inner_cv,
                n_jobs=-1,
                refit=True
            )
            grid.fit(X[tr_idx], y[tr_idx])
            model = grid.best_estimator_
            best_params = grid.best_params_
            print(f"[Fold {fold}] Best Params: {best_params}")
        else:
            model = pipeline
            model.fit(X[tr_idx], y[tr_idx])
            best_params = {"svc__C": 10, "svc__gamma": "scale"}

        preds = model.predict(X[te_idx])
        probs = model.predict_proba(X[te_idx])

        acc = accuracy_score(y[te_idx], preds)
        prec = precision_score(y[te_idx], preds, average="weighted", zero_division=0)
        rec = recall_score(y[te_idx], preds, average="weighted", zero_division=0)
        f1 = f1_score(y[te_idx], preds, average="weighted", zero_division=0)
        ll = log_loss(y[te_idx], probs)

        print(f"\n--- Fold {fold} ---")
        print(f"Accuracy: {acc:.6f}")
        print(f"Precision: {prec:.6f}")
        print(f"Recall: {rec:.6f}")
        print(f"F1: {f1:.6f}")
        print(f"Log Loss: {ll:.6f}")

        rows.append({
            "Fold": fold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "LogLoss": ll,
            "Best_C": best_params["svc__C"],
            "Best_Gamma": best_params["svc__gamma"]
        })

    df = pd.DataFrame(rows)

    # Compute averages
    avg_values = df[["Accuracy", "Precision", "Recall", "F1", "LogLoss"]].mean().to_dict()
    avg_values["Fold"] = "AVERAGE"
    avg_values["Best_C"] = "-"
    avg_values["Best_Gamma"] = "-"

    df_avg = pd.DataFrame([avg_values])
    df = pd.concat([df, df_avg], ignore_index=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    csv_path = os.path.join(MODEL_DIR, "cv_detailed_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n========== AVERAGE METRICS ==========")
    print(df.tail(1).to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\nTOTAL EXECUTION TIME: {elapsed:.2f} seconds")
    print(f"[OK] Saved CSV to {csv_path}")
