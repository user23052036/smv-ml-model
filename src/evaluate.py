# src/evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error
)

from extract_features import extract_features
from utils import load_model

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# ================= LABEL INFERENCE =================
def infer_labels_from_path(path: str):
    p = path.lower()

    freshness = "rotten" if "rotten" in p else "fresh"

    if "apple" in p:
        fruit = "apple"
    elif "banana" in p:
        fruit = "banana"
    elif "orange" in p:
        fruit = "orange"
    else:
        return None, None

    return fruit, freshness

# ================= LOAD TEST DATA =================
def load_test_data():
    X, y_fruit, y_fresh = [], [], []

    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if not f.endswith(IMAGE_EXTS):
                continue

            img_path = os.path.join(root, f)
            fruit, freshness = infer_labels_from_path(img_path)

            if fruit is None:
                continue

            try:
                feats, _ = extract_features(img_path)
                X.append(feats)
                y_fruit.append(fruit)
                y_fresh.append(1 if freshness == "fresh" else 0)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")

    if len(X) == 0:
        raise RuntimeError("No valid test samples found.")

    # Sanity check: feature lengths
    lengths = {len(v) for v in X}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent feature lengths: {lengths}")

    return np.vstack(X).astype(np.float32), np.array(y_fruit), np.array(y_fresh)

# ================= CONFUSION MATRIX SAVE =================
def save_confusion_matrix(cm, labels, title, out_path):
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Saved {out_path}")

# ================= MAIN =================
if __name__ == "__main__":

    print("[INFO] Loading test data...")
    X, y_fruit, y_fresh = load_test_data()
    print(f"[INFO] Test samples: {len(X)}")

    # Load models
    fruit_model = load_model(os.path.join(MODEL_DIR, "fruit_type_svm.joblib"))
    fresh_model = load_model(os.path.join(MODEL_DIR, "freshness_svm.joblib"))
    label_encoder = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # =====================================================
    # FRUIT CLASSIFICATION
    # =====================================================
    y_fruit_enc = label_encoder.transform(y_fruit)
    fruit_preds = fruit_model.predict(X)

    print("\n=== Fruit Classification Report ===")
    print(classification_report(y_fruit_enc, fruit_preds,
                                target_names=label_encoder.classes_))

    acc = accuracy_score(y_fruit_enc, fruit_preds)
    prec = precision_score(y_fruit_enc, fruit_preds, average="weighted")
    rec = recall_score(y_fruit_enc, fruit_preds, average="weighted")
    mse = mean_squared_error(y_fruit_enc, fruit_preds)

    print("[METRICS] Fruit")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"Sensitivity : {rec:.4f}")
    print(f"MSE         : {mse:.4f}")

    cm_fruit = confusion_matrix(y_fruit_enc, fruit_preds)
    save_confusion_matrix(
        cm_fruit,
        label_encoder.classes_,
        "Fruit Classification Confusion Matrix",
        os.path.join(MODEL_DIR, "confusion_matrix_fruit.png")
    )

    # =====================================================
    # FRESHNESS CLASSIFICATION
    # =====================================================
    fresh_preds = fresh_model.predict(X)

    print("\n=== Freshness Classification Report ===")
    print(classification_report(y_fresh, fresh_preds,
                                target_names=["rotten", "fresh"]))

    acc = accuracy_score(y_fresh, fresh_preds)
    prec = precision_score(y_fresh, fresh_preds)
    rec = recall_score(y_fresh, fresh_preds)
    mse = mean_squared_error(y_fresh, fresh_preds)

    print("[METRICS] Freshness")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"Sensitivity : {rec:.4f}")
    print(f"MSE         : {mse:.4f}")

    cm_fresh = confusion_matrix(y_fresh, fresh_preds)
    save_confusion_matrix(
        cm_fresh,
        ["rotten", "fresh"],
        "Freshness Classification Confusion Matrix",
        os.path.join(MODEL_DIR, "confusion_matrix_freshness.png")
    )
