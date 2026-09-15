"""Experimental three-point triangle proposals; no calibrated confidence.

The broad search band and leftward convergence are explicit scene priors.
The right boundary is maximum visible spread, NOT a detected roller nip line.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, grey_opening, grey_closing


def transform(points, matrix):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0, 0, 1]])
    q = np.c_[points, np.ones(len(points))] @ matrix.T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-10):
        raise ValueError('Invalid coordinate projection')
    return q[:, :2] / q[:, 2:]


def measure(vertices, contour=None):
    apex, upper, lower = np.asarray(vertices, np.float64)
    base = lower - upper
    width = float(np.linalg.norm(base))
    u, v = upper - apex, lower - apex
    lengths = [float(np.linalg.norm(u)), float(np.linalg.norm(v))]
    if min(width, *lengths) <= 1e-8:
        raise ValueError('Degenerate triangle')
    height = float(abs(np.linalg.det(np.stack([base, apex-upper]))) / width)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(u, v) / np.prod(lengths), -1, 1))))
    return dict(base_width=width, perpendicular_height=height,
                upper_side_length=lengths[0], lower_side_length=lengths[1],
                apex_angle_deg=angle, triangle_area=width*height/2,
                triangle_perimeter=width+sum(lengths))


def ridge_features(bgr):
    """All descriptors are computed on native ROI pixels, never enhanced previews."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    soft = cv2.GaussianBlur(gray, (0, 0), 1.0)
    residual = np.maximum(soft-cv2.GaussianBlur(soft, (0, 0), 9), 0)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1, ksize=3)
    xx = cv2.GaussianBlur(gx*gx, (0, 0), 3)
    yy = cv2.GaussianBlur(gy*gy, (0, 0), 3)
    xy = cv2.GaussianBlur(gx*gy, (0, 0), 3)
    coherence = np.sqrt((xx-yy)**2+4*xy**2) / (xx+yy+1e-6)
    horizontal = yy / (xx+yy+1e-6)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    features = np.dstack([lab, residual, coherence, horizontal])
    return features, residual


def candidate_triangle(score, valid, threshold_factor=1.0):
    h, w = score.shape
    # This rectangle is a scene-specific search prior, not a labeled triangle.
    x0, x1 = round(.14*w), round(.77*w)
    y0, y1 = round(.29*h), round(.78*h)
    band = score[y0:y1, x0:x1]
    support = valid[y0:y1, x0:x1]
    if support.sum() < 100:
        return None
    values = band[support]
    quantized = np.uint8(np.clip(values / max(float(values.max()), 1e-6), 0, 1)*255)
    otsu, _ = cv2.threshold(quantized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshold = max(float(otsu/255*values.max()), .002)
    # Exposure/contrast changes strongly across the transparent guard. A global
    # threshold alone discards the faint fibers on the right of almost every frame.
    # Anchor the vertical band at the strongest leftward fiber bundle to reduce
    # contamination from the separate bright machine lip above it.
    seed_profile = np.mean(band[:, :max(8, int(.1*len(band.T)))], axis=1)
    seed_row = int(np.argmax(gaussian_filter1d(seed_profile, 2)))
    support = support.copy()
    support[:max(0, seed_row-12), :] = False
    local_max = np.max(np.where(support, band, 0), axis=0, keepdims=True)
    local_threshold = np.maximum(.003, np.minimum(threshold, .3*local_max))*threshold_factor
    weights = np.where(support & (band > local_threshold), band, 0)
    upper, lower, supported = [], [], []
    for col in weights.T:
        total = col.sum()
        supported.append(total > threshold*2)
        if total <= threshold*2:
            upper.append(np.nan)
            lower.append(np.nan)
            continue
        cumulative = np.cumsum(col) / total
        upper.append(float(np.searchsorted(cumulative, .08)+y0))
        lower.append(float(np.searchsorted(cumulative, .92)+y0))
    supported = np.array(supported)
    if supported.mean() < .55:
        return None
    xs = np.arange(x0, x1)
    top = gaussian_filter1d(np.interp(xs, xs[supported], np.array(upper)[supported]), 5)
    bottom = gaussian_filter1d(np.interp(xs, xs[supported], np.array(lower)[supported]), 5)
    # Spatial envelope prior bridges short missing ridge responses. This is not
    # recovered image detail and is reported as geometric regularization.
    top = gaussian_filter1d(grey_opening(top, size=31), 5)
    bottom = gaussian_filter1d(grey_closing(bottom, size=31), 5)
    widths = bottom-top
    # Find the narrow end before the fan widens. Never extrapolate outside the image.
    end = int(.4*len(xs))
    apex_i = int(np.argmin(widths[:end]))
    # Maximum visible spread is an operational proxy, not a nip-line detection.
    base_start = max(apex_i+int(.2*w), int(.4*len(xs)))
    if base_start >= len(xs):
        return None
    base_i = base_start+int(np.argmax(widths[base_start:]))
    if widths[base_i] < 12 or xs[base_i]-xs[apex_i] < w*.15:
        return None
    apex = [float(xs[apex_i]), float((top[apex_i]+bottom[apex_i])/2)]
    vertices = np.array([apex, [xs[base_i], top[base_i]], [xs[base_i], bottom[base_i]]], np.float64)
    # The proposal target is exactly A-B-C; no curved contour is retained.
    contour = vertices.copy()
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 255)
    if np.any((mask > 0) & ~valid):
        return None
    return dict(vertices=vertices, contour=contour, mask=mask,
                supported_column_fraction=float(supported.mean()), threshold=threshold,
                tip_band_width_px=float(widths[apex_i]),
                search_rect=[x0, y0, x1, y1])
