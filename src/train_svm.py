# src/train_svm.py
import os
import glob
import numpy as np
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import hinge_loss

from extract_features import extract_features
from utils import save_model

DATA_DIR = "dataset/train"
MODEL_DIR = "models"


def parse_folder(folder):
    folder = folder.lower()
    if folder in ["apple", "banana", "orange"]:
        return folder, "fresh"
    if folder.startswith("rotten"):
        fruit = folder.replace("rotten", "")
        if fruit.endswith("s"):
            fruit = fruit[:-1]
        return fruit, "rotten"
    return None, None


def get_images(path):
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"):
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X        : (n_samples, n_features) float32
        y_fruit  : fruit labels (str)
        y_fresh  : freshness labels (1 = fresh, 0 = rotten)
    """
    X, y_fruit, y_fresh = [], [], []

    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        fruit, freshness = parse_folder(folder)
        if fruit is None:
            continue

        for img in get_images(folder_path):
            try:
                feats, _ = extract_features(img)
                X.append(feats)
                y_fruit.append(fruit)
                y_fresh.append(1 if freshness == "fresh" else 0)
            except Exception as e:
                print(f"[WARN] Skipping {img}: {e}")

    if not X:
        raise RuntimeError("No valid training samples found.")

    # sanity check
    lengths = {f.shape[0] for f in X}
    if len(lengths) != 1:
        raise RuntimeError(f"Feature length mismatch: {lengths}")

    X = np.vstack(X).astype(np.float32)
    return X, np.array(y_fruit), np.array(y_fresh)


if __name__ == "__main__":
    print("[INFO] Loading training data...")
    X, y_fruit, y_fresh = load_dataset()
    print("[INFO] Samples loaded:", len(X))

    os.makedirs(MODEL_DIR, exist_ok=True)

    # =======================
    # Encode fruit labels
    # =======================
    le = LabelEncoder()
    y_fruit_enc = le.fit_transform(y_fruit)
    save_model(le, f"{MODEL_DIR}/label_encoder.joblib")

    # =======================
    # Hyperparameter grid
    # =======================
    params = {
        "svc__C": [1, 10],
        "svc__gamma": ["scale", "auto"]
    }

    # =======================
    # Fruit classifier (SVM)
    # =======================
    print("[INFO] Training fruit classifier...")
    fruit_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])

    fruit_clf = GridSearchCV(
        fruit_model,
        params,
        cv=5,
        n_jobs=-1
    )
    fruit_clf.fit(X, y_fruit_enc)
    save_model(
        fruit_clf.best_estimator_,
        f"{MODEL_DIR}/fruit_type_svm.joblib"
    )

    # ============================================================
    # SVM SOLVER CONVERGENCE (max_iter vs hinge loss) – FRESHNESS
    # ============================================================
    print("[INFO] Tracking SVM solver convergence (max_iter vs hinge loss)...")

    MAX_ITERS = [50, 100, 300, 600, 1000]
    hinge_losses = []

    # hinge loss expects labels in {-1, +1}
    y_signed = np.where(y_fresh == 1, 1, -1)

    for it in MAX_ITERS:
        temp_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                max_iter=it
            ))
        ])

        temp_model.fit(X, y_fresh)

        X_scaled = temp_model.named_steps["scaler"].transform(X)
        scores = temp_model.named_steps["svc"].decision_function(X_scaled)

        loss = np.mean(np.maximum(0, 1 - y_signed * scores))
        hinge_losses.append(loss)

        print(f"[ITER {it}] Hinge Loss: {loss:.4f}")

    np.savez(
        f"{MODEL_DIR}/svm_convergence.npz",
        max_iter=np.array(MAX_ITERS),
        hinge_loss=np.array(hinge_losses)
    )

    # =======================
    # Final freshness model
    # =======================
    print("[INFO] Training freshness classifier...")
    fresh_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])

    fresh_clf = GridSearchCV(
        fresh_model,
        params,
        cv=5,
        n_jobs=-1
    )
    fresh_clf.fit(X, y_fresh)
    save_model(
        fresh_clf.best_estimator_,
        f"{MODEL_DIR}/freshness_svm.joblib"
    )

    print("[SUCCESS] Training complete.")
