# src/extract_features.py
"""
Feature extraction for fruit images.

Produces a feature vector (default 30 features) with the following order:
- RGB statistics: R_mean, G_mean, B_mean, R_std, G_std, B_std           (6)
- HSV statistics: H_mean_deg, S_mean, V_mean, H_std_deg, S_std, V_std  (6)
- LAB statistics: L_mean, L_std, a_mean, a_std, b_mean, b_std         (6)
- Texture: laplacian_variance, glcm_contrast, glcm_energy,
           glcm_homogeneity, grayscale_entropy                       (5)
- Shape: contour_area, perimeter, circularity, solidity,
         aspect_ratio, extent                                        (6)
- Decay indicator: dark_pixel_ratio                                  (1)

optional_extra=True appends 6 extra features:
R_median, G_median, B_median, mean_gradient_magnitude, H_skew, S_skew

Dependencies:
    opencv-python, numpy, scikit-image, scipy

Usage:
    fv, names = extract_features("path/to/img.jpg", resize_width=300, optional_extra=False)
"""
from typing import Tuple, List
import cv2
import numpy as np
from math import sqrt, pi, atan2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# -------------------- low-level helpers --------------------

def _load_image(path_or_array):
    """Return BGR uint8 image."""
    if isinstance(path_or_array, str):
        img = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path_or_array}")
        return img
    else:
        img = np.asarray(path_or_array)
        if img.dtype != np.uint8:
            img = (255 * (img.astype(np.float32) / np.max(img))).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

def _resize_keep_aspect(img, width):
    h, w = img.shape[:2]
    if w == 0 or width is None:
        return img
    if w == width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

def _get_mask_otsu(img_bgr):
    """Simple segmentation: blur -> otsu on gray -> take largest contour -> closed mask."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If object is darker than background, invert to prefer bright object
    # We'll pick the largest external contour either way.
    contours, _ = cv2.findContours(thr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # try inverted threshold / fallback
        _, thr2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thr2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # final fallback: whole image
            return np.ones_like(gray, dtype=np.uint8) * 255
    # choose largest contour
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    # morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def _masked_stats(channel: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    vals = channel[mask > 0]
    if vals.size == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std())

def _masked_mean(channel: np.ndarray, mask: np.ndarray) -> float:
    vals = channel[mask > 0]
    return float(vals.mean()) if vals.size else 0.0

def _circular_mean_std_deg(deg_vals: np.ndarray) -> Tuple[float, float]:
    """deg_vals: 0..360 degrees array"""
    if deg_vals.size == 0:
        return 0.0, 0.0
    rad = np.deg2rad(deg_vals)
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))
    mean_angle = atan2(sin_mean, cos_mean)
    mean_deg = (np.rad2deg(mean_angle) + 360.0) % 360.0
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    if R <= 0:
        circ_std = 0.0
    else:
        circ_std = float(np.sqrt(-2.0 * np.log(R)) * (180.0 / np.pi))
    return mean_deg, circ_std

# -------------------- texture & GLCM --------------------

def _laplacian_variance(gray: np.ndarray, mask: np.ndarray) -> float:
    g = gray.copy()
    # neutralize background by setting it to mean to avoid edge artifacts from borders
    g_masked = g.copy()
    mean_gray = float(g_masked[mask > 0].mean()) if np.any(mask > 0) else float(g_masked.mean())
    g_masked[mask == 0] = mean_gray
    lap = cv2.Laplacian(g_masked, cv2.CV_64F)
    return float(lap.var())

def _glcm_features(gray: np.ndarray, mask: np.ndarray, levels: int = 16):
    """
    Compute GLCM contrast, energy, homogeneity.
    gray must be 0-255 uint8. We quantize to `levels`.
    """
    if not np.any(mask > 0):
        return 0.0, 0.0, 0.0
    # quantize
    step = max(1, 256 // levels)
    quant = (gray // step).astype(np.uint8)
    # zero out background (will slightly bias, but simpler)
    quant_bg = quant.copy()
    quant_bg[mask == 0] = 0
    # compute GLCM with common angles
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    distances = [1]
    glcm = graycomatrix(quant_bg, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, "contrast").mean())
    energy = float(graycoprops(glcm, "energy").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    return contrast, energy, homogeneity

def _grayscale_entropy(gray: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask > 0):
        return 0.0
    vals = gray[mask > 0]
    return float(shannon_entropy(vals))

# -------------------- geometry --------------------

def _contour_geometry(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area": 0.0, "perimeter": 0.0, "circularity": 0.0,
            "solidity": 0.0, "aspect_ratio": 0.0, "extent": 0.0
        }
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = float(cv2.arcLength(c, True))
    circularity = float(4.0 * pi * area / (perimeter**2)) if perimeter > 0 and area > 0 else 0.0
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    extent = float(area) / (w * h) if (w * h) > 0 else 0.0
    return {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "extent": extent
    }

def _dark_pixel_ratio(gray: np.ndarray, mask: np.ndarray, factor: float = 0.6) -> float:
    if not np.any(mask > 0):
        return 0.0
    vals = gray[mask > 0]
    thresh = float(vals.mean()) * factor
    dark_count = int(np.sum(vals < thresh))
    return float(dark_count) / float(vals.size)

# -------------------- optional extras --------------------

def _median_rgb(bgr: np.ndarray, mask: np.ndarray):
    b, g, r = cv2.split(bgr)
    def m(ch):
        vals = ch[mask > 0]
        return float(np.median(vals)) if vals.size else 0.0
    return m(r), m(g), m(b)

def _mean_gradient_magnitude(gray: np.ndarray, mask: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    vals = grad[mask > 0]
    return float(vals.mean()) if vals.size else 0.0

def _skewness(arr: np.ndarray):
    if arr.size < 2:
        return 0.0
    m = arr.mean()
    std = arr.std(ddof=0)
    if std == 0:
        return 0.0
    return float(((arr - m) ** 3).mean() / (std ** 3))

# -------------------- public function --------------------

def extract_features(img_input,
                     resize_width: int = 300,
                     quant_levels: int = 16,
                     optional_extra: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from a fruit image.

    Returns:
        fv: numpy array (float32) feature vector
        names: list of feature names (same order as fv)
    """
    img_bgr = _load_image(img_input)
    img_bgr = _resize_keep_aspect(img_bgr, resize_width)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    mask = _get_mask_otsu(img_bgr)

    features = []
    names = []

    # --- RGB stats (means + stds) ---
    b, g, r = cv2.split(img_bgr)
    Rmean, Rstd = _masked_stats(r, mask)
    Gmean, Gstd = _masked_stats(g, mask)
    Bmean, Bstd = _masked_stats(b, mask)
    names += ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
    features += [Rmean, Gmean, Bmean, Rstd, Gstd, Bstd]

    # --- HSV stats (circular for H) ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # convert OpenCV H (0..179) to degrees 0..360
    Hdeg = (H.astype(np.float32) * 2.0)
    Hvals = Hdeg[mask > 0].flatten()
    Svals = S[mask > 0].astype(np.float32).flatten()
    Vvals = V[mask > 0].astype(np.float32).flatten()
    Hmean, Hstd = _circular_mean_std_deg(Hvals) if Hvals.size else (0.0, 0.0)
    Smean = float(Svals.mean()) if Svals.size else 0.0
    Sstd = float(Svals.std()) if Svals.size else 0.0
    Vmean = float(Vvals.mean()) if Vvals.size else 0.0
    Vstd = float(Vvals.std()) if Vvals.size else 0.0
    names += ["H_mean_deg", "S_mean", "V_mean", "H_std_deg", "S_std", "V_std"]
    features += [Hmean, Smean, Vmean, Hstd, Sstd, Vstd]

    # --- LAB stats ---
    Lch, ach, bch = cv2.split(lab)
    Lmean, Lstd = _masked_stats(Lch, mask)
    amean, astd = _masked_stats(ach, mask)
    bmean, bstd = _masked_stats(bch, mask)
    names += ["L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std"]
    features += [Lmean, Lstd, amean, astd, bmean, bstd]

    # --- Texture & surface quality ---
    lap_var = _laplacian_variance(gray, mask)
    glcm_contrast, glcm_energy, glcm_homog = _glcm_features(gray, mask, levels=quant_levels)
    entropy = _grayscale_entropy(gray, mask)
    names += ["laplacian_variance", "glcm_contrast", "glcm_energy", "glcm_homogeneity", "grayscale_entropy"]
    features += [lap_var, glcm_contrast, glcm_energy, glcm_homog, entropy]

    # --- Shape & geometry ---
    geom = _contour_geometry(mask)
    names += ["contour_area", "perimeter", "circularity", "solidity", "aspect_ratio", "extent"]
    features += [geom["area"], geom["perimeter"], geom["circularity"], geom["solidity"], geom["aspect_ratio"], geom["extent"]]

    # --- Decay indicator ---
    dpr = _dark_pixel_ratio(gray, mask, factor=0.6)
    names += ["dark_pixel_ratio"]
    features += [dpr]

    # optional extras (6 features)
    if optional_extra:
        rmed, gmed, bmed = _median_rgb(img_bgr, mask)
        grad_mean = _mean_gradient_magnitude(gray, mask)
        # skew for H and S (H in degrees)
        Hvals_for_skew = Hdeg[mask > 0].flatten()
        Svals_for_skew = S[mask > 0].astype(np.float32).flatten()
        h_skew = _skewness(Hvals_for_skew)
        s_skew = _skewness(Svals_for_skew)
        names += ["R_median", "G_median", "B_median", "mean_gradient_magnitude", "H_skew", "S_skew"]
        features += [rmed, gmed, bmed, grad_mean, h_skew, s_skew]

    fv = np.array(features, dtype=np.float32)
    return fv, names

# -------------------- quick CLI test --------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <image_path> [--extras]")
        sys.exit(1)
    path = sys.argv[1]
    extras = ("--extras" in sys.argv[2:]) or ("-e" in sys.argv[2:])
    fv, names = extract_features(path, resize_width=300, optional_extra=extras)
    print(f"Extracted {len(fv)} features (extras={extras})")
    for n, v in zip(names, fv):
        print(f"{n}: {v}")
