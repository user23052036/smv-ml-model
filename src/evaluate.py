import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from extract_features import extract_features
from utils import load_model

DATA_DIR = "dataset/test"
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
    X, y_fruit, y_fresh = [], [], []

    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if not f.endswith(IMAGE_EXTS):
                continue

            img_path = os.path.join(root, f)
            fruit, freshness = infer_labels_from_path(img_path)
            if fruit is None:
                continue

            feats = extract_features(img_path)
            X.append(feats)
            y_fruit.append(fruit)
            y_fresh.append(1 if freshness == "fresh" else 0)

    return np.array(X), np.array(y_fruit), np.array(y_fresh)

if __name__ == "__main__":
    X, y_fruit, y_fresh = load_test_data()

    fruit_model = load_model("models/fruit_type_svm.joblib")
    fresh_model = load_model("models/freshness_svm.joblib")
    le = load_model("models/label_encoder.joblib")

    y_fruit_enc = le.transform(y_fruit)

    print("\n=== Fruit Classification ===")
    fruit_preds = fruit_model.predict(X)
    print(classification_report(y_fruit_enc, fruit_preds, target_names=le.classes_))
    print(confusion_matrix(y_fruit_enc, fruit_preds))

    print("\n=== Freshness Classification ===")
    fresh_preds = fresh_model.predict(X)
    print(classification_report(y_fresh, fresh_preds))
    print(confusion_matrix(y_fresh, fresh_preds))
