# src/learning_curve.py

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from train_svm import load_dataset

MODEL_DIR = "models"

if __name__ == "__main__":

    print("[INFO] Loading dataset...")
    X, y_fruit, _ = load_dataset()

    if X.size == 0:
        raise RuntimeError("Dataset empty")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_fruit)

    # Ensure correct dtypes (important)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"[INFO] Samples: {len(X)}")

    # ❗ IMPORTANT: never include 1.0
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

    train_acc = []
    val_acc = []

    for frac in fractions:
        print(f"[INFO] Training with {int(frac*100)}% data")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            train_size=frac,
            stratify=y,
            random_state=42
        )

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10, gamma="scale"))
        ])

        model.fit(X_tr, y_tr)

        tr_score = model.score(X_tr, y_tr)
        val_score = model.score(X_val, y_val)

        train_acc.append(tr_score)
        val_acc.append(val_score)

        print(f"  Train acc: {tr_score:.4f} | Val acc: {val_score:.4f}")

    # ===== Plot =====
    os.makedirs(MODEL_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(fractions, train_acc, marker="o", label="Training Accuracy")
    plt.plot(fractions, val_acc, marker="s", label="Validation Accuracy")
    plt.xlabel("Fraction of Training Data Used")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve (SVM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(MODEL_DIR, "learning_curve.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved learning curve to {out_path}")
