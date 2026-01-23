
# Fruit Freshness Detection (SVM)

A compact, terminal-first project that classifies fruit type (`apple`, `banana`, `orange`) and predicts freshness (`fresh` / `rotten`) using handcrafted computer-vision features and Support Vector Machines (SVM). The system is designed to run locally (Linux) using Python and provides CLI utilities for training, evaluation, and prediction.

---

## Highlights

* Lightweight, CPU-friendly pipeline — no deep learning required
* Interpretable 30-dimensional handcrafted feature vector
* Two SVM classifiers:

  * Fruit type classifier (apple / banana / orange)
  * Freshness classifier (fresh / rotten)
* Reproducible training, evaluation, and CLI prediction

---

## Status / Artifacts

* Trained models:

  * `models/fruit_type_svm.joblib`
  * `models/freshness_svm.joblib`
  * `models/label_encoder.joblib`
* Frozen feature schema:

  * `models/feature_schema.json`
* Evaluation outputs:

  * `models/confusion_matrix_fruit.png`
  * `models/confusion_matrix_freshness.png`

### Confusion matrices

**Fruit classification**

![Fruit Confusion Matrix](models/confusion_matrix_fruit.png)

**Freshness classification**

![Freshness Confusion Matrix](models/confusion_matrix_freshness.png)

---

## Project layout

```
mini-project/
├── dataset/
│     ├── train/
│     └── test/
|
├── models/
│     ├── feature_schema.json
│     ├── fruit_type_svm.joblib
│     |── freshness_svm.joblib
│     ├── label_encoder.joblib
│     └── confusion_matrix_*.png
|
├── src/
│     ├── extract_features.py
│     ├── train_svm.py
│     ├── evaluate.py
│     ├── predict_cli.py
│     ├── save_feature_schema.py
│     └── utils.py
|
├── requirements.txt
└── README.md
```

> The loader supports both flat folder naming (e.g. `rottenapples`) and hierarchical naming (`apple/rotten`) as long as fruit and freshness keywords are present.

---

## Setup

From the project root:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required dependencies include:

* numpy
* opencv-python
* scikit-learn
* scikit-image
* scipy
* joblib
* matplotlib

---

## Training

Train both classifiers (fruit type and freshness):

```
python src/train_svm.py
```

What happens internally:

* Feature extraction from `dataset/train`
* Label encoding for fruit classes
* GridSearchCV over SVM (RBF kernel) with StandardScaler
* Best models saved to `models/`

---

## Evaluation

Run evaluation on the test split:

```
python src/evaluate.py
```

Outputs:

* Precision / recall / F1-score reports
* Confusion matrices printed to console
* Confusion matrix images saved for documentation

---

## Prediction (CLI)

Predict a single image:

```
python src/predict_cli.py --image /full/path/to/image.jpg
```

Example:

Analyzing image: /home/user/orange.jpg
Fruit: orange (88.41%)
Freshness score: 76.92% → FRESH

Freshness threshold defaults to 50% and can be adjusted in `predict_cli.py`.

---

## Feature extraction (frozen schema)

The system uses a fixed 30-feature vector. The exact order is saved in `models/feature_schema.json` and must not be changed without retraining.

Feature categories:

* **Color**

  * RGB mean & standard deviation
  * HSV circular mean & std
  * LAB mean & std
* **Texture**

  * Laplacian variance
  * GLCM contrast, energy, homogeneity
  * Grayscale entropy
* **Shape**

  * Area, perimeter
  * Circularity, solidity
  * Aspect ratio, extent
* **Decay**

  * Dark pixel ratio

This design balances interpretability and performance without deep learning.

---

## Segmentation approach

* Convert to grayscale
* Gaussian blur
* Otsu thresholding
* Largest contour selected as fruit mask

If segmentation fails, the extractor falls back to whole-image statistics (lower confidence, but no crash).

---

## Performance (current split)

* Fruit classification accuracy: ~98%
* Freshness classification accuracy: ~96%

Results depend on dataset quality and lighting conditions.

---

## Limitations & failure modes

* Background similarity can break segmentation
* Strong lighting color casts affect color features
* Very early decay may be visually indistinguishable
* Dataset leakage can inflate metrics if not careful

These limitations are documented and expected for classical CV systems.

---
