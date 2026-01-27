# src/train_svm.py
import os
import glob
import numpy as np
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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
    Load training dataset.
    
    Returns:
        X: Feature matrix of shape (n_samples, n_features) with dtype float32
        y_fruit: Fruit labels as strings
        y_fresh: Freshness labels as integers (1 for fresh, 0 for rotten)
    """
    X, y_fruit, y_fresh = [], [], []

    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        fruit, freshness = parse_folder(folder)
        if fruit is None:
            continue

        images = get_images(folder_path)
        for img in images:
            try:
                feats, _ = extract_features(img)          # <-- unpack tuple (fv, names)
                X.append(feats)
                y_fruit.append(fruit)
                y_fresh.append(1 if freshness == "fresh" else 0)
            except Exception as e:
                print(f"[WARN] Skipping {img}: {e}")

    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])

    # Validate all feature lengths are identical
    lengths = [f.shape[0] for f in X]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Feature length mismatch across samples: {set(lengths)}. "
                           "Ensure extract_features returns consistent length for all images.")

    X_arr = np.vstack(X).astype(np.float32)
    return X_arr, np.array(y_fruit), np.array(y_fresh)

if __name__ == "__main__":
    print("[INFO] Loading training data...")
    X, y_fruit, y_fresh = load_dataset()
    print("[INFO] Samples loaded:", 0 if X.size == 0 else len(X))

    if X.size == 0:
        raise RuntimeError("No training images found or no valid features extracted. Check dataset folders and extractor.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    le = LabelEncoder()
    y_fruit_enc = le.fit_transform(y_fruit)
    save_model(le, f"{MODEL_DIR}/label_encoder.joblib")

    params = {
        "svc__C": [1, 10],
        "svc__gamma": ["scale", "auto"]
    }

    print("[INFO] Training fruit classifier...")
    fruit_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])
    fruit_clf = GridSearchCV(fruit_model, params, cv=5, n_jobs=-1)
    fruit_clf.fit(X, y_fruit_enc)
    save_model(fruit_clf.best_estimator_, f"{MODEL_DIR}/fruit_type_svm.joblib")

    print("[INFO] Training freshness classifier...")
    fresh_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])
    fresh_clf = GridSearchCV(fresh_model, params, cv=5, n_jobs=-1)
    fresh_clf.fit(X, y_fresh)
    save_model(fresh_clf.best_estimator_, f"{MODEL_DIR}/freshness_svm.joblib")

    print("[SUCCESS] Training complete.")
