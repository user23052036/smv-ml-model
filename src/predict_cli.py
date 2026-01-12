import argparse
import os
import numpy as np
from extract_features import extract_features
from utils import load_model

# Always resolve paths from project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def predict(image_path):
    print(f"Analyzing image: {image_path}")
    print("-" * 60)

    label_encoder = load_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    fruit_model = load_model(os.path.join(MODEL_DIR, "fruit_type_svm.joblib"))
    fresh_model = load_model(os.path.join(MODEL_DIR, "freshness_svm.joblib"))

    features = extract_features(image_path).reshape(1, -1)

    # Fruit prediction
    fruit_probs = fruit_model.predict_proba(features)[0]
    fruit_idx = np.argmax(fruit_probs)
    fruit_name = label_encoder.inverse_transform([fruit_idx])[0]
    fruit_conf = fruit_probs[fruit_idx] * 100

    # Freshness prediction
    fresh_prob = fresh_model.predict_proba(features)[0][1] * 100
    verdict = "FRESH" if fresh_prob >= 50 else "ROTTEN"

    print(f"Fruit: {fruit_name} ({fruit_conf:.2f}%)")
    print(f"Freshness score: {fresh_prob:.2f}% -> {verdict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    predict(args.image)
