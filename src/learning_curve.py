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

    # ============================================================
    # PART 1: STANDARD LEARNING CURVE (data fraction vs accuracy)
    # ============================================================
    print("[INFO] Loading dataset...")
    X, y_fruit, _ = load_dataset()

    if X.size == 0:
        raise RuntimeError("Dataset empty")

    le = LabelEncoder()
    y = le.fit_transform(y_fruit)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"[INFO] Samples: {len(X)}")

    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

    train_acc = []
    val_acc = []

    for frac in fractions:
        print(f"[INFO] Training with {int(frac * 100)}% data")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
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

    lc_path = os.path.join(MODEL_DIR, "learning_curve.png")
    plt.savefig(lc_path, dpi=200)
    plt.close()

    print(f"[OK] Saved learning curve to {lc_path}")

    # ============================================================
    # PART 2: SVM SOLVER CONVERGENCE (max_iter vs hinge loss)
    # ============================================================
    conv_path = os.path.join(MODEL_DIR, "svm_convergence.npz")

    if not os.path.exists(conv_path):
        print("[WARN] svm_convergence.npz not found. Skipping convergence plot.")
    else:
        print("[INFO] Plotting SVM solver convergence...")

        data = np.load(conv_path)
        max_iter = data["max_iter"]
        hinge_loss = data["hinge_loss"]

        plt.figure(figsize=(8, 5))
        plt.plot(max_iter, hinge_loss, marker="o")
        plt.xlabel("SVM max_iter")
        plt.ylabel("Hinge Loss")
        plt.title("SVM Solver Convergence (max_iter vs hinge loss)")
        plt.grid(True)
        plt.tight_layout()

        out_path = os.path.join(MODEL_DIR, "svm_convergence.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Saved SVM convergence plot to {out_path}")
