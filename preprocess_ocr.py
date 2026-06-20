import cv2
import numpy as np

def preprocess_plate(image):
    """
    Professor requirements:
    1. Grayscale
    2. Adaptive Thresholding  
    3. Morphological Operations
    4. Contour Detection
    5. Perspective Correction
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 2. Adaptive Thresholding (Gaussian)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 3. Morphological Operations (Close then Open)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Contour Detection for Perspective Correction
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Minimum area rectangle (handles rotation)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Get width and height
        width, height = rect[1]
        if width < height:
            width, height = height, width
        
        # Perspective correction if size is reasonable
        if width > 30 and height > 10:
            # Order points: top-left, top-right, bottom-right, bottom-left
            pts = np.zeros((4, 2), dtype=np.float32)
            s = box.sum(axis=1)
            pts[0] = box[np.argmin(s)]  # Top-left
            pts[2] = box[np.argmax(s)]  # Bottom-right
            
            diff = np.diff(box, axis=1)
            pts[1] = box[np.argmin(diff)]  # Top-right
            pts[3] = box[np.argmax(diff)]  # Bottom-left
            
            # Destination points
            dst = np.array([
                [0, 0],
                [width-1, 0],
                [width-1, height-1],
                [0, height-1]
            ], dtype=np.float32)
            
            # 5. Perspective Correction
            try:
                M = cv2.getPerspectiveTransform(pts, dst)
                corrected = cv2.warpPerspective(morph, M, (int(width), int(height)))
                return cv2.resize(corrected, (128, 32))
            except:
                pass
    
    # Fallback: just resize
    return cv2.resize(morph, (128, 32))
