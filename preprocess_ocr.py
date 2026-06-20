import cv2
import numpy as np

# Final OCR input size used by your CRNN
OCR_W = 128
OCR_H = 32

def order_points(pts):
    """Order 4 points: tl, tr, br, bl"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # tl
    rect[2] = pts[np.argmax(s)]   # br
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # tr
    rect[3] = pts[np.argmax(d)]   # bl
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    maxW = max(maxW, 1)
    maxH = max(maxH, 1)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped

def preprocess_plate(plate_bgr, return_debug=False):
    """
    Full professor-required preprocessing pipeline:
    1) Grayscale
    2) Adaptive threshold
    3) Morphological operations
    4) Contour detection
    5) Perspective correction

    Returns:
      processed (uint8, OCR_H x OCR_W grayscale)
      + optional debug dict
    """
    debug = {}

    # Safety fallback input
    if plate_bgr is None or plate_bgr.size == 0:
        fallback = np.zeros((OCR_H, OCR_W), dtype=np.uint8)
        return (fallback, debug) if return_debug else fallback

    # 1) Grayscale
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    debug["gray"] = gray.copy()

    # light denoise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    debug["blur"] = blur.copy()

    # 2) Adaptive threshold
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )
    debug["adaptive_thresh"] = th.copy()

    # 3) Morphological ops
    # open -> remove tiny noise
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, k_open, iterations=1)

    # close -> connect character strokes
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close, iterations=1)
    debug["morph"] = morph.copy()

    # 4) Contour detection (find dominant 4-corner region)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0
    h, w = gray.shape[:2]
    img_area = h * w

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area:   # ignore tiny regions
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 5) Perspective correction needs 4 points
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2).astype(np.float32)

    # Use perspective correction if good quad found
    if best_quad is not None:
        warped_gray = four_point_transform(gray, best_quad)
        debug["perspective_used"] = True
        debug["quad"] = best_quad.copy()
    else:
        # fallback: no perspective correction possible
        warped_gray = gray
        debug["perspective_used"] = False
        debug["quad"] = None

    # final OCR resize
    processed = cv2.resize(warped_gray, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
    debug["final"] = processed.copy()

    if return_debug:
        return processed, debug
    return processed
