# src/cross_validation.py

import numpy as np
from typing import Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# reuse dataset loader safely
from train_svm import load_dataset

K = 5

if __name__ == "__main__":

    print("[INFO] Loading training data...")
    X, y_fruit, _ = load_dataset()

    if X.size == 0:
        raise RuntimeError("No training data found.")

    # Ensure X is properly typed as float32 array
    X = np.asarray(X, dtype=np.float32)
    
    # Encode labels (REQUIRED)
    le = LabelEncoder()
    y = le.fit_transform(y_fruit)
    
    # Ensure y is properly typed as integer array
    y = np.asarray(y, dtype=np.int64)

    print(f"[INFO] Samples: {len(X)}")
    print(f"[INFO] Classes: {list(le.classes_)}")

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    scores = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=10, gamma="scale"))
        ])

        model.fit(X[tr], y[tr])
        preds = model.predict(X[te])

        acc = accuracy_score(y[te], preds)
        scores.append(acc)

        print(f"Fold {fold}: accuracy = {acc:.4f}")

    print("\n=== K-Fold Cross Validation Results ===")
    print(f"Mean accuracy: {np.mean(scores):.4f}")
    print(f"Std accuracy : {np.std(scores):.4f}")
