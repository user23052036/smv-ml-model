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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


class FruitFreshnessClassifier:
    """Two-stage SVM-based fruit freshness classifier."""
    
    def __init__(self):
        """Initialize the classifier with default parameters."""
        self.fruit_type_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        self.fruit_type_scaler = StandardScaler()

        self.fruit_models = {
            "apple":  {"model": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True), "scaler": StandardScaler()},
            "banana": {"model": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True), "scaler": StandardScaler()},
            "orange": {"model": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True), "scaler": StandardScaler()}
        }

        self.fruit_type_label_map = {"apple": 0, "banana": 1, "orange": 2}
        self.freshness_label_map = {"rotten": 0, "fresh": 1}

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

    def analyze_banana_color(self, image_path: str) -> Tuple[float, float]:
        """
        Analyze banana image color distribution for relabeling.

        Args:
            image_path: Path to the banana image

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

    def relabel_banana_image(self, image_path: str, original_label: int) -> int:
        """
        Apply programmatic relabeling for banana images based on color analysis.

        Args:
            image_path: Path to the banana image
            original_label: Original label from folder name (0=rotten, 1=fresh)

        Returns:
            Relabeled value: 0=rotten, 1=fresh
        """
        green_percentage, yellow_percentage = self.analyze_banana_color(image_path)

        # Apply relabeling rules
        if green_percentage > 15:
            # If significant green pixels, label as fresh
            return 1
        elif yellow_percentage > 40 and green_percentage < 5:
            # If mostly yellow with minimal green, treat as fresh (overripe but edible)
            return 1
        else:
            # Otherwise, label as rotten
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

        # Iterate through all subdirectories
        for folder_path in dataset_path.iterdir():
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
        
        # Scale features for fruit type classification
        X_train_ft_scaled = self.fruit_type_scaler.fit_transform(X_train_ft)
        X_test_ft_scaled = self.fruit_type_scaler.transform(X_test_ft)
        
        # Train fruit type model
        self.fruit_type_model.fit(X_train_ft_scaled, y_train_ft)
        
        # Evaluate fruit type model
        y_pred_ft = self.fruit_type_model.predict(X_test_ft_scaled)
        fruit_type_accuracy = accuracy_score(y_test_ft, y_pred_ft)
        
        print(f"Fruit type classification accuracy: {fruit_type_accuracy:.4f} ({fruit_type_accuracy*100:.2f}%)")
        
        # Stage 2: Train Fruit-Specific Freshness Classifiers
        print("\n" + "="*50)
        print("STAGE 2: Training Fruit-Specific Freshness Classifiers")
        print("="*50)

        accuracies = {}
        for fruit_type in self.fruit_models.keys():
            fruit_label = self.fruit_type_label_map[fruit_type]
            mask = y_fruit_type == fruit_label

            if np.any(mask):
                print(f"\nTraining {fruit_type.capitalize()} Freshness Model...")
                X_fruit = X[mask]
                y_fruit = y_freshness[mask].copy()

                # Apply banana relabeling ONLY for bananas
                if fruit_type == "banana":
                    print("Applying programmatic banana relabeling...")
                    fruit_indices = np.where(mask)[0]
                    relabeled_count = 0
                    for i, idx in enumerate(fruit_indices):
                        image_path = image_paths[idx]
                        original_label = y_freshness[idx]
                        new_label = self.relabel_banana_image(image_path, original_label)
                        y_fruit[i] = new_label
                        if new_label != original_label:
                            relabeled_count += 1
                    print(f"Relabeled {relabeled_count} banana images based on color analysis")

                X_train, X_test, y_train, y_test = train_test_split(
                    X_fruit, y_fruit, test_size=0.2, stratify=y_fruit, random_state=42
                )

                X_train_scaled = self.fruit_models[fruit_type]["scaler"].fit_transform(X_train)
                X_test_scaled = self.fruit_models[fruit_type]["scaler"].transform(X_test)

                self.fruit_models[fruit_type]["model"].fit(X_train_scaled, y_train)

                y_pred = self.fruit_models[fruit_type]["model"].predict(X_test_scaled)
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
            "fruit_type": {
                "model": self.fruit_type_model,
                "scaler": self.fruit_type_scaler,
                "label_map": self.fruit_type_label_map
            },
            "freshness": {
                "apple": {
                    "model": self.fruit_models["apple"]["model"],
                    "scaler": self.fruit_models["apple"]["scaler"]
                },
                "banana": {
                    "model": self.fruit_models["banana"]["model"],
                    "scaler": self.fruit_models["banana"]["scaler"]
                },
                "orange": {
                    "model": self.fruit_models["orange"]["model"],
                    "scaler": self.fruit_models["orange"]["scaler"]
                }
            },
            "freshness_label_map": self.freshness_label_map
        }

        joblib.dump(models_data, model_path)
        print(f"\nAll models saved to: {model_path}")
        
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

        # Load fruit type model
        self.fruit_type_model = models_data["fruit_type"]["model"]
        self.fruit_type_scaler = models_data["fruit_type"]["scaler"]
        self.fruit_type_label_map = models_data["fruit_type"]["label_map"]

        # Load freshness models into dictionary structure
        self.fruit_models["apple"]["model"] = models_data["freshness"]["apple"]["model"]
        self.fruit_models["apple"]["scaler"] = models_data["freshness"]["apple"]["scaler"]

        self.fruit_models["banana"]["model"] = models_data["freshness"]["banana"]["model"]
        self.fruit_models["banana"]["scaler"] = models_data["freshness"]["banana"]["scaler"]

        self.fruit_models["orange"]["model"] = models_data["freshness"]["orange"]["model"]
        self.fruit_models["orange"]["scaler"] = models_data["freshness"]["orange"]["scaler"]

        self.freshness_label_map = models_data["freshness_label_map"]

        self.is_trained = True
        print(f"All models loaded from: {model_path}")
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict freshness of a single image using two-stage approach.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_class, freshness_percentage)
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained. Run training first.")
        
        # Extract features
        feature = self.extract_features(image_path)
        feature = feature.reshape(1, -1)
        
        # Stage 1: Predict fruit type
        feature_scaled_ft = self.fruit_type_scaler.transform(feature)
        fruit_type_prediction = self.fruit_type_model.predict(feature_scaled_ft)[0]
        fruit_type_probabilities = self.fruit_type_model.predict_proba(feature_scaled_ft)[0]
        
        # Get fruit type name
        fruit_type_names = {v: k for k, v in self.fruit_type_label_map.items()}
        predicted_fruit_type = fruit_type_names[fruit_type_prediction]
        
        print(f"Fruit type: {predicted_fruit_type}")
        
        # Stage 2: Predict freshness using fruit-specific model
        feature_scaled = self.fruit_models[predicted_fruit_type]["scaler"].transform(feature)
        freshness_model = self.fruit_models[predicted_fruit_type]["model"]
        
        # Predict freshness
        freshness_prediction = freshness_model.predict(feature_scaled)[0]
        freshness_probabilities = freshness_model.predict_proba(feature_scaled)[0]
        
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
        default="./original_data_set",
        help="Path to dataset directory (default: ./original_data_set)"
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
            predicted_class, freshness_percentage = classifier.predict(args.image)
            
            print(f"Predicted: {predicted_class}")
            print(f"Freshness: {freshness_percentage:.2f} %")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
