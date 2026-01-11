#!/usr/bin/env python3
"""
Quick Inference Test Script for Fruit Freshness Classifier

Loads the trained model and runs inference with debug output on sample images.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import run module
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import FruitFreshnessClassifier


def main():
    """Run quick inference tests."""
    classifier = FruitFreshnessClassifier()

    # Load the trained model
    try:
        classifier.load_models("./models/svm_freshness.joblib")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Sample images to test
    images = [
        "~/Desktop/banana_fresh.jpg",
        "~/Desktop/banana_2.jpg",
        "~/Desktop/raw_banana.jpg",
        "~/Desktop/orange.jpg"
    ]

    print("=== Quick Inference Test ===")

    results = []
    fruit_type_probas = []
    freshness_probas = []

    for img_path in images:
        img_full_path = os.path.expanduser(img_path)
        print(f"\n--- Testing {img_full_path} ---")

        try:
            # Run prediction
            predicted_class, freshness_percentage = classifier.predict(img_full_path, debug=False)

            print(f"Final Result: {predicted_class} ({freshness_percentage:.2f}%)")

            results.append((predicted_class, freshness_percentage))

            # Collect probabilities for validation
            feature = classifier.extract_features(img_full_path).reshape(1, -1)

            ft_proba = classifier.fruit_type_model.predict_proba(feature)[0]
            fruit_type_probas.append(tuple(ft_proba))

            ft_class = classifier.fruit_type_model.predict(feature)[0]
            fruit_names = {v: k for k, v in classifier.fruit_type_label_map.items()}
            fruit_type = fruit_names[ft_class]

            fresh_proba = classifier.fruit_models[fruit_type].predict_proba(feature)[0]
            freshness_probas.append(tuple(fresh_proba))

        except Exception as e:
            print(f"Error processing {img_full_path}: {e}")
            sys.exit(1)

    # Validation logic
    print("\n=== Validation Results ===")

    # Check if fruit type probabilities vary
    unique_ft_probas = set(fruit_type_probas)
    if len(unique_ft_probas) <= 1:
        print("FAIL: Fruit type predict_proba outputs are identical or nearly identical!")
        print(f"All fruit type probas: {fruit_type_probas}")
        sys.exit(1)
    else:
        print(f"PASS: Fruit type predict_proba vary ({len(unique_ft_probas)} unique values)")

    # Check if freshness probabilities vary
    unique_fresh_probas = set(freshness_probas)
    if len(unique_fresh_probas) <= 1:
        print("FAIL: Freshness predict_proba outputs are identical or nearly identical!")
        print(f"All freshness probas: {freshness_probas}")
        sys.exit(1)
    else:
        print(f"PASS: Freshness predict_proba vary ({len(unique_fresh_probas)} unique values)")

    # Check if predictions are reasonable (at least some variation)
    classes = [r[0] for r in results]
    percentages = [r[1] for r in results]

    if len(set(classes)) <= 1:
        print("WARNING: All predictions have the same class")
    else:
        print(f"PASS: Predictions vary across classes: {set(classes)}")

    pct_range = max(percentages) - min(percentages)
    if pct_range < 1.0:
        print(f"WARNING: Freshness percentages vary by only {pct_range:.2f}%")
    else:
        print(f"PASS: Freshness percentages vary by {pct_range:.2f}%")

    print("\n=== All Tests Passed ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
