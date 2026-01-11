# Fruit Freshness Classifier

A minimal terminal-based machine learning prototype for classifying fruit freshness using Support Vector Machine (SVM) with color histogram features.

## Overview

This project automatically classifies fruits as **fresh** or **rotten** using a trained SVM model. The classifier analyzes color histograms from fruit images to determine freshness levels.

## Features

- **Simple & Fast**: Minimal dependencies, no GUI, no notebooks
- **Automatic Labeling**: Fresh/rotten labels determined from folder names
- **Color + Texture Classification**: Uses HSV color histograms and LBP texture features for improved accuracy
- **Two-Stage Classification**: First identifies fruit type, then uses fruit-specific freshness models for better accuracy
- **Intelligent Banana Labeling**: Banana freshness labels are adjusted programmatically to align with human perception
- **Terminal Interface**: Clean command-line interface with train and predict commands
- **Model Persistence**: Trained models saved and loaded using joblib

## Project Structure

```
MINI-PROJECT/
├── original_data_set/        # Existing dataset (DO NOT MODIFY)
├── run.py                    # Main entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Auto-created, stores trained model
└── .venv/                    # Virtual environment
```

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the SVM model using your dataset:

```bash
python run.py train
```

**Optional parameters:**
- `--dataset`: Path to dataset directory (default: `./original_data_set`)
- `--model`: Path to save trained model (default: `./models/svm_freshness.joblib`)

**Example:**
```bash
python run.py train --dataset ./my_dataset --model ./models/my_model.joblib
```

**Output:**
```
Loading dataset from: ./original_data_set
Processing fresh images from: freshapples
  Loaded 200 images from freshapples
Processing fresh images from: freshbanana
  Loaded 200 images from freshbanana
Processing fresh images from: freshoranges
  Loaded 200 images from freshoranges
Processing rotten images from: rottenapples
  Loaded 200 images from rottenapples
Processing rotten images from: rottenbanana
  Loaded 200 images from rottenbanana
Processing rotten images from: rottenoranges
  Loaded 200 images from rottenoranges

Dataset loaded successfully:
  Total images: 1200
  Fresh: 600
  Rotten: 600
  Unknown: 0

Starting training process...

Train/Test split:
  Training samples: 960
  Test samples: 240

Scaling features...
Training SVM model...

Training completed!
Test accuracy: 0.9458 (94.58%)
Model saved to: ./models/svm_freshness.joblib

==================================================
TRAINING SUMMARY
==================================================
Total images processed: 1200
Fresh images: 600
Rotten images: 600
Training samples: 960
Test samples: 240
Test accuracy: 0.9458 (94.58%)
==================================================
```

### Making Predictions

Predict the freshness of a single image:

```bash
python run.py predict --image ./path/to/your/image.jpg
```

**Optional parameters:**
- `--image`: Path to the image file to classify (required)
- `--model`: Path to trained model (default: `./models/svm_freshness.joblib`)
- `--debug`: Enable detailed debug output (default: False)

**Example:**
```bash
python run.py predict --image ./test_images/apple.jpg --model ./models/my_model.joblib
```

**Output:**
```
Model loaded from: ./models/svm_freshness.joblib
Predicted: fresh
Freshness: 87.45 %
```

### Debug Mode

Use the `--debug` flag for detailed inference information:

```bash
python run.py predict --image ./test_images/apple.jpg --debug
```

**Debug output includes:**
- Absolute image path and read status
- Feature vector fingerprint (first 8 values, statistics, L2 norm)
- Fruit-type model predictions and probabilities
- Freshness model predictions and probabilities
- Label map verification

### Running Diagnostics

Run comprehensive diagnostics on sample images to detect issues:

```bash
python run.py run_diagnostics
```

**What it does:**
- Tests inference on 4 sample images with debug output
- Compares feature vectors between all image pairs
- Warns if feature vectors are nearly identical (L2 distance < 1e-6)
- Helps identify problems with feature extraction or constant predictions

**Sample images tested:**
- `~/Desktop/banana_fresh.jpg`
- `~/Desktop/banana_2.jpg`
- `~/Desktop/raw_banana.jpg`
- `~/Desktop/orange.jpg`

## Dataset Requirements

### Folder Structure

The dataset should be organized with subfolders containing images:

```
dataset_root/
├── freshapples/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── freshbanana/
├── freshoranges/
├── rottenapples/
├── rottenbanana/
└── rottenoranges/
```

### Label Assignment

- **Fresh**: Any folder name containing "fresh" (case-insensitive) → label = 1
- **Rotten**: Any folder name containing "rotten" (case-insensitive) → label = 0
- **Unknown**: Folders not matching above patterns are skipped with a warning

### Image Requirements

- Supported formats: PNG, JPG, JPEG (any format supported by OpenCV)
- Images are automatically resized to 100x100 pixels
- Color space converted from BGR to HSV for feature extraction
- Unreadable images are skipped safely

## Technical Details

### Feature Extraction

1. **Image Processing**:
   - Read image using OpenCV
   - Resize to 100x100 pixels
   - Convert BGR → HSV color space

2. **Feature Extraction**:
   - Extract 3D color histogram
   - Bins: (8, 8, 8) for H, S, V channels
   - Normalize histogram
   - Flatten to feature vector

### Model Architecture

- **Classifier**: `sklearn.svm.SVC`
  - Kernel: RBF
  - C: 1.0
  - Gamma: "scale"
  - Probability: True

- **Preprocessing**:
  - StandardScaler for feature normalization
  - Train/test split: 80%/20% with stratification

- **Model Storage**:
  ```python
  {
    "model": svm_model,
    "scaler": scaler,
    "label_map": {"rotten": 0, "fresh": 1}
  }
  ```

### Freshness Percentage

The freshness percentage represents the model's confidence that the fruit is fresh:

```
Freshness % = Probability(fresh) × 100
```

- **0-50%**: Model predicts rotten (lower values = more confident rotten)
- **50-100%**: Model predicts fresh (higher values = more confident fresh)
- **~50%**: Model is uncertain

## Error Handling

- **Model not found**: "Model not trained. Run train first."
- **Image cannot be read**: Clear error message with file path
- **Unknown folders**: Warning message, folder skipped
- **No valid images**: Error with helpful message

## Dependencies

- `numpy`: Numerical operations
- `opencv-python`: Image processing and feature extraction
- `scikit-learn`: SVM implementation and preprocessing
- `joblib`: Model serialization

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure virtual environment is activated and dependencies installed
2. **Permission errors**: Check file permissions for dataset and model directories
3. **Memory issues**: For very large datasets, consider processing in batches
4. **Poor accuracy**: Verify dataset quality and ensure proper fresh/rotten labeling

### Dataset Problems

- **Mixed labels in folders**: Ensure each folder contains only fresh OR rotten images
- **Corrupted images**: Use image validation tools to check dataset integrity
- **Imbalanced classes**: Model handles stratification automatically

## Performance Notes

- **Training time**: Depends on dataset size, typically seconds to minutes
- **Prediction time**: Very fast, milliseconds per image
- **Memory usage**: Scales with dataset size during training, minimal during prediction
- **Accuracy**: Depends on dataset quality, typically 85-95% for well-labeled datasets

## License

This project is provided as-is for educational and prototyping purposes.
