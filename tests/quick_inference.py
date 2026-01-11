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
    classifier.load_models("./models/svm_freshness.joblib")

    # Sample images to test
    images = [
        "~/Desktop/banana_fresh.jpg",
        "~/Desktop/banana_2.jpg",
        "~/Desktop/raw_banana.jpg",
        "~/Desktop/orange.jpg"
    ]

    print("=== Quick Inference Test ===")

    for img_path in images:
        img_full_path = os.path.expanduser(img_path)
        print(f"\n--- Testing {img_full_path} ---")

        try:
            # Run prediction with debug output
            predicted_class, freshness_percentage = classifier.predict(img_full_path, debug=True)

            print(f"Final Result: {predicted_class} ({freshness_percentage:.2f}%)")

        except Exception as e:
            print(f"Error processing {img_full_path}: {e}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
