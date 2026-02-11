import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import log_loss

from train_svm import load_dataset

MODEL_DIR = "models"

# Load data
X, _, y = load_dataset()

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_losses = []
folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_val)
    loss = log_loss(y_val, probs)

    log_losses.append(loss)
    folds.append(fold)

    print(f"Fold {fold}: Log Loss = {loss:.4f}")

# Plot
plt.figure(figsize=(7, 4))
plt.plot(folds, log_losses, marker="o")
plt.xlabel("Fold")
plt.ylabel("Log Loss")
plt.title("Log Loss vs Fold (SVM)")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{MODEL_DIR}/logloss_vs_fold.png", dpi=200)
plt.close()

print(f"[OK] Saved models/logloss_vs_fold.png")
