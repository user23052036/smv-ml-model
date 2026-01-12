# Fruit Freshness Detection (SVM) — README

A compact, terminal-first project that classifies fruits (**apple / banana / orange**) and predicts **freshness (fresh / rotten)** using classical computer-vision features and Support Vector Machines (SVM). Designed to run locally (Linux) with a simple CLI.

---

## Quick status

* Models: saved to `models/` after training (`fruit_type_svm.joblib`, `freshness_svm.joblib`, `label_encoder.joblib`)
* CLI: `src/predict_cli.py` — predict a single image from the terminal
* Eval: `src/evaluate.py` — test set metrics and confusion matrices
* Core feature extractor: `src/extract_features.py`

---

## Project layout

```
mini-project/
├── dataset/
│   ├── train/
│   │   ├── apple/            # OR freshapples/  (both formats supported)
│   │   ├── banana/
│   │   ├── orange/
│   │   ├── rottenapples/
│   │   ├── rottenbanana/
│   │   └── rottenoranges/
│   └── test/                 # same rules as train/
├── models/                   # auto-created by training
├── src/
│   ├── extract_features.py
│   ├── train_svm.py
│   ├── evaluate.py
│   ├── predict_cli.py
│   └── utils.py
├── requirements.txt
└── README.md
```

> Note: The loader accepts either the **flat** folder convention (`freshapples`, `rottenapples`) or the **hierarchical** one (`apple/fresh`, `apple/rotten`). Both are handled by the training scripts.

---

## Getting started (copy-paste)

```bash
# from project root
cd ~/Desktop/mini-project

# create and activate venv (if not already)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

`requirements.txt` should include at least:

```
numpy
opencv-python
scikit-learn
joblib
tqdm
matplotlib
```

---

## Train models

```bash
# trains both fruit-type and freshness SVMs (saves to models/)
python src/train_svm.py
```

What happens:

* Extracts features from every image in `dataset/train`
* Encodes fruit labels
* Runs GridSearchCV (C=[1,10], gamma=['scale','auto']) with StandardScaler + SVC
* Saves `models/fruit_type_svm.joblib`, `models/freshness_svm.joblib`, `models/label_encoder.joblib`

---

## Evaluate on test set

```bash
python src/evaluate.py
```

Output:

* Classification reports (precision/recall/f1)
* Confusion matrices for fruit and freshness

---

## Predict single image (CLI)

```bash
python src/predict_cli.py --image /full/path/to/image.jpg
```

Example output:

```
Analyzing image: /home/midori/Desktop/orange.jpg
------------------------------------------------------------
Fruit: orange (88.41%)
Freshness score: 76.92% -> FRESH
```

Rules:

* Fresh if freshness probability ≥ 50% (adjustable in `predict_cli.py`)
* Predictions include probabilities for transparency

---

## Feature extractor (what it computes)

From `src/extract_features.py`:

* LAB color stats (mean, std for L, a, b) — 6 features
* BGR means (B, G, R) — 3 features
* Mean Laplacian (texture) — 1 feature
* Equivalent diameter (from largest contour area) — 1 feature
* Final vector length ≈ 11 floats

Segmentation: grayscale → Gaussian blur → Otsu threshold → largest contour (fallback: full image if segmentation fails).

---

## Supported dataset folder names

Your loader accepts:

* Flat naming: `freshapples`, `rottenapples`, `freshbanana`, `rottenbanana`, `freshoranges`, `rottenoranges`
* Or hierarchical: `apple/fresh`, `apple/rotten`, etc.

If you change folder names, ensure the name contains `fresh`/`rotten` or the fruit name (`apple`, `banana`, `orange`) so the parser can infer both labels.

---

## Troubleshooting

* `No training images found` → check image extensions (.jpg/.jpeg/.png). Scripts search common extensions.
* `Model file not found` → run training first; saved models live in `models/`.
* Low confidence / wrong predictions on web images:

  * Use images with plain background & full fruit
  * Consider background cropping or more training data
  * Add `CalibratedClassifierCV` for probability calibration if needed
* If segmentation fails, feature extractor falls back to the whole image.

---

## How to improve (quick path)

* Add more training images (best impact)
* Add simple augmentations (flip, rotate, color jitter)
* Add HSV features or GLCM texture for extra discriminative power
* Use per-fruit freshness classifiers (apple-specific, banana-specific)
* Apply probability calibration (Platt scaling / isotonic)

---

## License & citation

This work and the dataset (if you publish it) can be released under **CC0 1.0** (public domain) or choose another license in `dataset-metadata.json`.
If you use this dataset in a paper, cite as:

```
user2036. Fruit Freshness Dataset (Apple, Banana, Orange). Kaggle (year).
```

---

## Contact / contributions

* If you find a bug or want assistance, open an issue in the project repo or message `user2036` on Kaggle.
* Pull requests and improvements (feature additions, better segmentation, calibration) are welcome.

---

## Short checklist before submission / demo

* [ ] Models trained and saved to `models/`
* [ ] `evaluate.py` shows acceptable accuracy on `dataset/test`
* [ ] README updated (this file)
* [ ] `dataset/` structured and referenced correctly in `dataset-metadata.json`
* [ ] If publishing dataset: confirm `dataset-metadata.json` and set dataset to **Public** on Kaggle

---

If you want, I will now:

* produce a **short abstract (3–4 lines)** for the Kaggle description (summary)
* or a **one-page project report** for submission (PDF-ready)

Tell me which and I’ll output it exactly.
