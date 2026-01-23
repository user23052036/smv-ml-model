# src/evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from extract_features import extract_features
from utils import load_model

# Resolve paths safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

def infer_labels_from_path(path):
    path = path.lower()
    freshness = "rotten" if "rotten" in path else "fresh"

    if "apple" in path:
        fruit = "apple"
    elif "banana" in path:
        fruit = "banana"
    elif "orange" in path:
        fruit = "orange"
    else:
        return None, None

    return fruit, freshness

def load_test_data():
    X_list, y_fruit, y_fresh = [], [], []

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
                X_list.append(feats)
                y_fruit.append(fruit)
                y_fresh.append(1 if freshness == "fresh" else 0)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")

    if len(X_list) == 0:
        raise RuntimeError("No valid test samples found.")

    lengths = {len(f) for f in X_list}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent feature lengths in test set: {lengths}")

    X = np.vstack(X_list).astype(np.float32)
    return X, np.array(y_fruit), np.array(y_fresh)

def save_confusion_matrix(cm, labels, title, filename):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"[OK] Saved {filename}")

if __name__ == "__main__":
    print("[INFO] Loading test data...")
    X, y_fruit, y_fresh = load_test_data()
    print(f"[INFO] Test samples: {len(X)}")

    fruit_model = load_model(os.path.join(MODEL_DIR, "fruit_type_svm.joblib"))
    fresh_model = load_model(os.path.join(MODEL_DIR, "freshness_svm.joblib"))
    le = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # === Fruit classification ===
    y_fruit_enc = le.transform(y_fruit)
    fruit_preds = fruit_model.predict(X)

    print("\n=== Fruit Classification ===")
    print(classification_report(y_fruit_enc, fruit_preds, target_names=le.classes_))
    cm_fruit = confusion_matrix(y_fruit_enc, fruit_preds)
    print(cm_fruit)

    save_confusion_matrix(
        cm_fruit,
        labels=le.classes_,
        title="Fruit Classification Confusion Matrix",
        filename=os.path.join(MODEL_DIR, "confusion_matrix_fruit.png")
    )

    # === Freshness classification ===
    fresh_preds = fresh_model.predict(X)

    print("\n=== Freshness Classification ===")
    print(classification_report(y_fresh, fresh_preds, target_names=["rotten", "fresh"]))
    cm_fresh = confusion_matrix(y_fresh, fresh_preds)
    print(cm_fresh)

    save_confusion_matrix(
        cm_fresh,
        labels=["rotten", "fresh"],
        title="Freshness Classification Confusion Matrix",
        filename=os.path.join(MODEL_DIR, "confusion_matrix_freshness.png")
    )
