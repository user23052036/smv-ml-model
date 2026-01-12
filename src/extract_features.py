"""
Feature extraction module for fruit images.
Extracts color (LAB + BGR), size (equivalent diameter), and texture (Laplacian) features.
"""
import cv2
import numpy as np
from math import sqrt, pi


def extract_features(img_path, resize_width=300):
    """
    Extract robust features from a fruit image.
    
    Features extracted:
    - Color: LAB channel statistics (mean & std of L, a, b)
    - Color: BGR channel means
    - Texture: Mean Laplacian (edge strength)
    - Size: Equivalent diameter from segmented fruit
    
    Args:
        img_path: Path to the input image
        resize_width: Standard width to resize images for consistent feature scales
        
    Returns:
        numpy array of features (length ~11)
    """
    # Read and validate image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    
    # Resize to standard width (maintains aspect ratio)
    h, w = img.shape[:2]
    if w > 0:
        scale = resize_width / float(w)
        img = cv2.resize(img, (resize_width, int(h * scale)))
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Segment fruit using Otsu thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
    
    # Find largest contour (assumed to be the fruit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: use entire image if segmentation fails
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        area = gray.shape[0] * gray.shape[1]
    else:
        # Use largest contour as fruit mask
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        area = cv2.contourArea(largest_contour)
    
    # Compute equivalent diameter (circle with same area)
    eq_diameter = sqrt(4 * area / pi) if area > 0 else 0
    
    # Extract color features from LAB color space (within mask)
    features = []
    for channel in cv2.split(lab):  # L, a, b channels
        masked_values = channel[mask == 255]
        if masked_values.size > 0:
            features.append(masked_values.mean())  # Mean
            features.append(masked_values.std())   # Standard deviation
        else:
            features.append(0)
            features.append(0)
    
    # Extract color features from BGR (within mask)
    for channel in cv2.split(img):  # B, G, R channels
        masked_values = channel[mask == 255]
        if masked_values.size > 0:
            features.append(masked_values.mean())
        else:
            features.append(0)
    
    # Texture feature: mean Laplacian (edge strength)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_masked = laplacian[mask == 255]
    if laplacian_masked.size > 0:
        features.append(np.abs(laplacian_masked).mean())
    else:
        features.append(0)
    
    # Add size feature
    features.append(eq_diameter)
    
    return np.array(features, dtype=float)


if __name__ == "__main__":
    # Test feature extraction on a sample image
    import sys
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        try:
            feats = extract_features(test_img)
            print(f"Extracted {len(feats)} features from {test_img}")
            print(f"Feature vector: {feats}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python extract_features.py <image_path>")
