#!/usr/bin/env python3
"""
Fruit Freshness Classifier using SVM

A minimal terminal-based ML prototype for classifying fruit freshness
using Support Vector Machine with color histogram features.

Usage:
    python run.py train          # Train the model
    python run.py predict --image <path>  # Predict freshness of an image
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


class FruitFreshnessClassifier:
    """Two-stage SVM-based fruit freshness classifier."""
    
    def __init__(self):
        """Initialize the classifier with default parameters."""
        self.fruit_type_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
        ])

        self.fruit_models = {
            "apple": Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
            ]),
            "banana": Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
            ]),
            "orange": Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
            ])
        }

        self.fruit_type_label_map = {"apple": 0, "banana": 1, "orange": 2}
        self.freshness_label_map = {"rotten": 0, "fresh": 1}

        self.COLOR_CORRECTION_ENABLED = {"banana": True}

        self.is_trained = False
    
    def extract_color_features(self, image_path: str) -> np.ndarray:
        """
        Extract HSV color histogram features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Flattened histogram feature vector
            
        Raises:
            ValueError: If image cannot be read
        """
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize to 100x100
        image = cv2.resize(image, (100, 100))
        
        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract 3D color histogram with 8 bins per channel
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        # Normalize histogram
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def extract_texture_features(self, image_path: str) -> np.ndarray:
        """
        Extract LBP texture histogram features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Flattened LBP histogram feature vector
            
        Raises:
            ValueError: If image cannot be read
        """
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Resize to 100x100
        image = cv2.resize(image, (100, 100))
        
        # Convert to grayscale for LBP
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute Local Binary Pattern
        lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
        
        # Extract histogram with 26 bins (0 to 25 for uniform LBP)
        lbp_hist = np.histogram(lbp.ravel(), bins=26, range=(0, 26))[0]
        
        # Normalize histogram
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Add small epsilon to avoid division by zero
        
        return lbp_hist
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract combined color and texture features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Combined feature vector (color + texture)
            
        Raises:
            ValueError: If image cannot be read
        """
        # Extract color features
        color_features = self.extract_color_features(image_path)
        
        # Extract texture features
        texture_features = self.extract_texture_features(image_path)
        
        # Combine features using horizontal stack
        combined_features = np.hstack([color_features, texture_features])
        
        return combined_features
    
    def get_fruit_type_from_folder(self, folder_name: str) -> str:
        """
        Determine fruit type from folder name.
        
        Args:
            folder_name: Name of the folder containing images
            
        Returns:
            Fruit type: "apple", "banana", "orange", or "unknown"
        """
        folder_lower = folder_name.lower()
        if "apple" in folder_lower:
            return "apple"
        elif "banana" in folder_lower:
            return "banana"
        elif "orange" in folder_lower:
            return "orange"
        else:
            return "unknown"
    
    def get_freshness_label_from_folder(self, folder_name: str) -> int:
        """
        Determine freshness label based on folder name.

        Args:
            folder_name: Name of the folder containing images

        Returns:
            1 for fresh, 0 for rotten, -1 for unknown
        """
        folder_lower = folder_name.lower()
        if "fresh" in folder_lower:
            return 1
        elif "rotten" in folder_lower:
            return 0
        else:
            return -1

    def analyze_color_distribution(self, image_path: str) -> Tuple[float, float]:
        """
        Analyze image color distribution for relabeling.

        Args:
            image_path: Path to the image

        Returns:
            Tuple of (green_percentage, yellow_percentage)
        """
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, 0.0

        # Resize to 100x100
        image = cv2.resize(image, (100, 100))

        # Convert BGR to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Count total pixels
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]

        # Count green pixels (H in [35, 85])
        green_mask = (hsv_image[:, :, 0] >= 35) & (hsv_image[:, :, 0] <= 85)
        green_pixels = np.sum(green_mask)

        # Count yellow pixels (H in [20, 35])
        yellow_mask = (hsv_image[:, :, 0] >= 20) & (hsv_image[:, :, 0] <= 35)
        yellow_pixels = np.sum(yellow_mask)

        # Calculate percentages
        green_percentage = (green_pixels / total_pixels) * 100
        yellow_percentage = (yellow_pixels / total_pixels) * 100

        return green_percentage, yellow_percentage

    def apply_color_based_label_correction(self, image_path: str, fruit_type: str, original_label: int) -> int:
        """
        Apply color-based label correction if enabled for the fruit type.

        Args:
            image_path: Path to the image
            fruit_type: Type of fruit ("apple", "banana", "orange")
            original_label: Original label from folder name (0=rotten, 1=fresh)

        Returns:
            Corrected label: 0=rotten, 1=fresh
        """
        if not self.COLOR_CORRECTION_ENABLED.get(fruit_type, False):
            return original_label

        green_percentage, yellow_percentage = self.analyze_color_distribution(image_path)

        # Apply relabeling rules
        if green_percentage > 15:
            return 1
        elif yellow_percentage > 40 and green_percentage < 5:
            return 1
        else:
            return 0
    
    def load_dataset(self, dataset_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, int], Dict[str, int]]:
        """
        Load and process the dataset from the given root directory.

        Args:
            dataset_root: Path to the dataset root directory

        Returns:
            X: Feature matrix
            fruit_type_labels: Fruit type labels (0=apple, 1=banana, 2=orange)
            freshness_labels: Freshness labels (0=rotten, 1=fresh)
            image_paths: List of image paths for relabeling
            fruit_type_counts: Dictionary with fruit type counts
            freshness_counts: Dictionary with freshness counts
        """
        print(f"Loading dataset from: {dataset_root}")

        features = []
        fruit_type_labels = []
        freshness_labels = []
        image_paths = []
        fruit_type_counts = {"apple": 0, "banana": 0, "orange": 0, "unknown": 0}
        freshness_counts = {"fresh": 0, "rotten": 0, "unknown": 0}

        dataset_path = Path(dataset_root)
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_root}")

        train_path = dataset_path / "train"
        if not train_path.exists():
            raise ValueError(f"Training directory not found: {train_path}")

        # Iterate through all subdirectories in train/
        for folder_path in train_path.iterdir():
            if not folder_path.is_dir():
                continue

            folder_name = folder_path.name
            fruit_type = self.get_fruit_type_from_folder(folder_name)
            freshness_label = self.get_freshness_label_from_folder(folder_name)

            if fruit_type == "unknown" or freshness_label == -1:
                print(f"Warning: Unknown folder '{folder_name}', skipping...")
                fruit_type_counts["unknown"] += 1
                freshness_counts["unknown"] += 1
                continue

            print(f"Processing {fruit_type} images from: {folder_name}")

            # Process all image files in the folder
            image_count = 0
            for image_path in folder_path.glob("*"):
                if image_path.is_file():
                    try:
                        feature = self.extract_features(str(image_path))
                        features.append(feature)
                        fruit_type_labels.append(self.fruit_type_label_map[fruit_type])
                        freshness_labels.append(freshness_label)
                        image_paths.append(str(image_path))
                        fruit_type_counts[fruit_type] += 1
                        freshness_counts["fresh" if freshness_label == 1 else "rotten"] += 1
                        image_count += 1
                    except ValueError as e:
                        print(f"  Warning: Skipping unreadable image {image_path.name}")
                        continue

            print(f"  Loaded {image_count} images from {folder_name}")

        if len(features) == 0:
            raise ValueError("No valid images found in dataset")

        X = np.array(features)
        y_fruit_type = np.array(fruit_type_labels)
        y_freshness = np.array(freshness_labels)

        print(f"\nDataset loaded successfully:")
        print(f"  Total images: {len(features)}")
        print(f"  Fruit types: {fruit_type_counts}")
        print(f"  Freshness: {freshness_counts}")

        return X, y_fruit_type, y_freshness, image_paths, fruit_type_counts, freshness_counts
    
    def train(self, dataset_root: str, model_path: str) -> Dict[str, Any]:
        """
        Train the two-stage SVM models on the dataset.
        
        Args:
            dataset_root: Path to the dataset root directory
            model_path: Path where to save the trained models
            
        Returns:
            Dictionary containing training results
        """
        print("Starting two-stage training process...")
        
        # Load dataset
        X, y_fruit_type, y_freshness, image_paths, fruit_type_counts, freshness_counts = self.load_dataset(dataset_root)
        
        # Stage 1: Train Fruit Type Classifier
        print("\n" + "="*50)
        print("STAGE 1: Training Fruit Type Classifier")
        print("="*50)
        
        # Split for fruit type classification
        X_train_ft, X_test_ft, y_train_ft, y_test_ft = train_test_split(
            X, y_fruit_type, test_size=0.2, stratify=y_fruit_type, random_state=42
        )

        print(f"Training samples: {len(X_train_ft)}")
        print(f"Test samples: {len(X_test_ft)}")

        # Train fruit type model (pipeline handles scaling)
        self.fruit_type_model.fit(X_train_ft, y_train_ft)

        # Evaluate fruit type model
        y_pred_ft = self.fruit_type_model.predict(X_test_ft)
        fruit_type_accuracy = accuracy_score(y_test_ft, y_pred_ft)
        
        print(f"Fruit type classification accuracy: {fruit_type_accuracy:.4f} ({fruit_type_accuracy*100:.2f}%)")
        
        # Stage 2: Train Fruit-Specific Freshness Classifiers
        print("\n" + "="*50)
        print("STAGE 2: Training Fruit-Specific Freshness Classifiers")
        print("="*50)

        accuracies = {}
        correction_counts = {}
        for fruit_type in self.fruit_models.keys():
            fruit_label = self.fruit_type_label_map[fruit_type]
            mask = y_fruit_type == fruit_label

            if np.any(mask):
                print(f"\nTraining {fruit_type.capitalize()} Freshness Model...")
                X_fruit = X[mask]
                y_fruit = y_freshness[mask].copy()

                # Apply color-based label corrections conditionally
                if self.COLOR_CORRECTION_ENABLED.get(fruit_type, False):
                    print(f"Applying color-based label corrections for {fruit_type}...")
                    fruit_indices = np.where(mask)[0]
                    relabeled_count = 0
                    for i, idx in enumerate(fruit_indices):
                        image_path = image_paths[idx]
                        original_label = y_freshness[idx]
                        new_label = self.apply_color_based_label_correction(image_path, fruit_type, original_label)
                        y_fruit[i] = new_label
                        if new_label != original_label:
                            relabeled_count += 1
                    print(f"Relabeled {relabeled_count} {fruit_type} images")
                    correction_counts[fruit_type] = relabeled_count

                X_train, X_test, y_train, y_test = train_test_split(
                    X_fruit, y_fruit, test_size=0.2, stratify=y_fruit, random_state=42
                )

                # Train freshness model (pipeline handles scaling)
                self.fruit_models[fruit_type].fit(X_train, y_train)

                y_pred = self.fruit_models[fruit_type].predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies[fruit_type] = accuracy
                print(f"{fruit_type.capitalize()} freshness accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            else:
                print(f"No {fruit_type} data found!")
                accuracies[fruit_type] = 0.0
        
        # Save all models in structured format
        model_dir = Path(model_path).parent
        model_dir.mkdir(exist_ok=True)

        models_data = {
            "fruit_type": self.fruit_type_model,
            "freshness": {
                "apple": self.fruit_models["apple"],
                "banana": self.fruit_models["banana"],
                "orange": self.fruit_models["orange"]
            },
            "fruit_type_label_map": self.fruit_type_label_map,
            "freshness_label_map": self.freshness_label_map
        }

        joblib.dump(models_data, model_path)
        print(f"\nAll models saved to: {model_path}")

        if correction_counts:
            print("\nLabel corrections applied:")
            for fruit, count in correction_counts.items():
                if count > 0:
                    print(f"  {fruit}: {count} samples")

        self.is_trained = True

        return {
            "total_images": len(X),
            "fruit_type_counts": fruit_type_counts,
            "freshness_counts": freshness_counts,
            "fruit_type_accuracy": fruit_type_accuracy,
            "apple_accuracy": accuracies.get("apple", 0.0),
            "banana_accuracy": accuracies.get("banana", 0.0),
            "orange_accuracy": accuracies.get("orange", 0.0)
        }
    
    def load_models(self, model_path: str):
        """Load all trained models from structured file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        models_data = joblib.load(model_path)

        # Load fruit type model (pipeline)
        self.fruit_type_model = models_data["fruit_type"]
        self.fruit_type_label_map = models_data["fruit_type_label_map"]

        # Load freshness models (pipelines)
        self.fruit_models["apple"] = models_data["freshness"]["apple"]
        self.fruit_models["banana"] = models_data["freshness"]["banana"]
        self.fruit_models["orange"] = models_data["freshness"]["orange"]

        self.freshness_label_map = models_data["freshness_label_map"]

        # Validate pipelines have predict_proba
        if not hasattr(self.fruit_type_model, 'predict_proba'):
            raise ValueError("Fruit type model missing predict_proba")
        for fruit in self.fruit_models:
            if not hasattr(self.fruit_models[fruit], 'predict_proba'):
                raise ValueError(f"{fruit} model missing predict_proba")

        self.is_trained = True
        print(f"All models loaded from: {model_path}")
    
    def predict(self, image_path: str, debug: bool = False) -> Tuple[str, float]:
        """
        Predict freshness of a single image using two-stage approach.

        Args:
            image_path: Path to the image file
            debug: If True, print detailed debug information

        Returns:
            Tuple of (predicted_class, freshness_percentage)
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained. Run training first.")

        # Extract features
        feature = self.extract_features(image_path)
        feature = feature.reshape(1, -1)

        if debug:
            print(f"Absolute image path: {os.path.abspath(image_path)}")
            img_check = cv2.imread(image_path)
            if img_check is None:
                print("cv2.imread() failed")
            else:
                print(f"cv2.imread() success, shape: {img_check.shape}")
            print(f"Feature fingerprint:")
            print(f"  First 8 values: {feature[0][:8]}")
            print(f"  Min: {feature.min():.6f}, Max: {feature.max():.6f}, Mean: {feature.mean():.6f}, Std: {feature.std():.6f}")
            print(f"  L2 norm: {np.linalg.norm(feature):.6f}")

        # Safety check: feature dimension
        expected_features = self.fruit_type_model.named_steps["scaler"].n_features_in_
        if feature.shape[1] != expected_features:
            raise ValueError(f"Feature dimension mismatch: got {feature.shape[1]}, expected {expected_features}")

        # Stage 1: Predict fruit type (pipeline handles scaling)
        try:
            fruit_type_prediction = self.fruit_type_model.predict(feature)[0]
            fruit_type_probabilities = self.fruit_type_model.predict_proba(feature)[0]
        except Exception as e:
            print(f"Error in fruit type prediction: {e}")
            raise

        if debug:
            print(f"Fruit-type model:")
            print(f"  predict(): {fruit_type_prediction}")
            print(f"  predict_proba(): {fruit_type_probabilities}")

        # Get fruit type name
        fruit_type_names = {v: k for k, v in self.fruit_type_label_map.items()}
        predicted_fruit_type = fruit_type_names[fruit_type_prediction]

        if debug:
            print(f"Resolved fruit type: {predicted_fruit_type}")
            print(f"fruit_type_label_map: {self.fruit_type_label_map}")

        print(f"Fruit type: {predicted_fruit_type}")

        # Stage 2: Predict freshness using fruit-specific model (pipeline handles scaling)
        try:
            freshness_prediction = self.fruit_models[predicted_fruit_type].predict(feature)[0]
            freshness_probabilities = self.fruit_models[predicted_fruit_type].predict_proba(feature)[0]
        except Exception as e:
            print(f"Error in freshness prediction: {e}")
            raise

        if debug:
            print(f"Freshness model:")
            print(f"  predict(): {freshness_prediction}")
            print(f"  predict_proba(): {freshness_probabilities}")

        # Get freshness probability (probability of being fresh)
        freshness_prob = freshness_probabilities[1]  # Index 1 corresponds to "fresh" class
        freshness_percentage = freshness_prob * 100

        predicted_class = "fresh" if freshness_prediction == 1 else "rotten"

        return predicted_class, freshness_percentage


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Fruit Freshness Classifier using SVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py train
  python run.py predict --image ./test_image.jpg
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--dataset",
        default="./dataset",
        help="Path to dataset directory (default: ./dataset)"
    )
    train_parser.add_argument(
        "--model", 
        default="./models/svm_freshness.joblib",
        help="Path to save the trained model (default: ./models/svm_freshness.joblib)"
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict freshness of an image")
    predict_parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to classify"
    )
    predict_parser.add_argument(
        "--model",
        default="./models/svm_freshness.joblib",
        help="Path to the trained model (default: ./models/svm_freshness.joblib)"
    )
    predict_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    # Run diagnostics command
    diagnostics_parser = subparsers.add_parser("run_diagnostics", help="Run diagnostics on sample images")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    classifier = FruitFreshnessClassifier()
    
    try:
        if args.command == "train":
            # Suppress sklearn warnings during training
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                
                results = classifier.train(args.dataset, args.model)
                
                print(f"\n" + "="*50)
                print("TWO-STAGE TRAINING SUMMARY")
                print("="*50)
                print(f"Total images processed: {results['total_images']}")
                print(f"Fruit type distribution: {results['fruit_type_counts']}")
                print(f"Freshness distribution: {results['freshness_counts']}")
                print(f"Fruit type accuracy: {results['fruit_type_accuracy']:.4f} ({results['fruit_type_accuracy']*100:.2f}%)")
                print(f"Apple freshness accuracy: {results['apple_accuracy']:.4f} ({results['apple_accuracy']*100:.2f}%)")
                print(f"Banana freshness accuracy: {results['banana_accuracy']:.4f} ({results['banana_accuracy']*100:.2f}%)")
                print(f"Orange freshness accuracy: {results['orange_accuracy']:.4f} ({results['orange_accuracy']*100:.2f}%)")
                print("="*50)
        
        elif args.command == "predict":
            # Load models
            classifier.load_models(args.model)

            # Make prediction
            predicted_class, freshness_percentage = classifier.predict(args.image, debug=getattr(args, 'debug', False))

            print(f"Predicted: {predicted_class}")
            print(f"Freshness: {freshness_percentage:.2f} %")

        elif args.command == "run_diagnostics":
            # Load models
            classifier.load_models("./models/svm_freshness.joblib")

            images = [
                os.path.expanduser("~/Desktop/banana_fresh.jpg"),
                os.path.expanduser("~/Desktop/banana_2.jpg"),
                os.path.expanduser("~/Desktop/raw_banana.jpg"),
                os.path.expanduser("~/Desktop/orange.jpg")
            ]

            features_list = []
            for img in images:
                print(f"\n=== Diagnostics for {img} ===")
                try:
                    predicted_class, freshness_percentage = classifier.predict(img, debug=True)
                    print(f"Predicted: {predicted_class}")
                    print(f"Freshness: {freshness_percentage:.2f} %")
                    # For simplicity, extract feature again for comparison
                    feature = classifier.extract_features(img)
                    features_list.append((img, feature))
                except Exception as e:
                    print(f"Error predicting {img}: {e}")

            # Compare features
            print(f"\n=== Feature Vector Comparisons ===")
            for i in range(len(features_list)):
                for j in range(i+1, len(features_list)):
                    img1, feat1 = features_list[i]
                    img2, feat2 = features_list[j]
                    l2_dist = np.linalg.norm(feat1 - feat2)
                    print(f"L2 distance between {os.path.basename(img1)} and {os.path.basename(img2)}: {l2_dist:.6f}")
                    if l2_dist < 1e-6:
                        print("WARNING: Feature vectors are nearly identical! Possible bug in feature extraction.")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
