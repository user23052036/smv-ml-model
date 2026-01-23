# src/save_feature_schema.py
import json
import os
import glob
from extract_features import extract_features

DATASET_DIR = "dataset/train"
OUTPUT_PATH = "models/feature_schema.json"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

def find_any_image(root):
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for ext in IMAGE_EXTS:
            files = glob.glob(os.path.join(folder_path, f"*{ext}"))
            if files:
                return files[0]
    return None

if __name__ == "__main__":
    sample_image = find_any_image(DATASET_DIR)

    if sample_image is None:
        raise RuntimeError("No training images found in dataset/train")

    print(f"[INFO] Using sample image: {sample_image}")

    features, names = extract_features(sample_image)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(names, f, indent=2)

    print(f"[OK] Feature schema saved to {OUTPUT_PATH}")
    print(f"[INFO] Total features: {len(names)}")
