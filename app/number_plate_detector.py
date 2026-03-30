
import cv2
import numpy as np
import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class NumberPlateDetector:
    """Detect and recognize vehicle number plates"""
    
    def __init__(self):
        self.plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$')
    
    def detect_plate(self, frame: np.ndarray, vehicle_bbox: tuple) -> Optional[Dict]:
        """Detect number plate in vehicle region"""
        x1, y1, x2, y2 = vehicle_bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular shapes (license plates)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:  # Typical plate area range
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    # Potential plate found
                    plate_roi = self._extract_plate_roi(roi, contour)
                    plate_text = self._recognize_text(plate_roi)
                    
                    if plate_text and self._validate_plate(plate_text):
                        return {
                            'plate_number': plate_text,
                            'confidence': 0.85,
                            'bbox': cv2.boundingRect(contour)
                        }
        
        return None
    
    def _extract_plate_roi(self, roi, contour):
        """Extract plate region"""
        x, y, w, h = cv2.boundingRect(contour)
        plate_roi = roi[y:y+h, x:x+w]
        return plate_roi
    
    def _recognize_text(self, plate_roi):
        """Recognize text from plate image (simulated)"""
        # In production, use OCR like Tesseract
        # For demo, return simulated plate numbers
        import random
        
        # Simulated Indian number plates
        plates = [
            "KA01AB1234", "MH02CD5678", "DL03EF9012", "TN04GH3456",
            "HR05IJ7890", "UP06KL1234", "GJ07MN5678", "RJ08OP9012"
        ]
        return random.choice(plates)
    
    def _validate_plate(self, plate_text: str) -> bool:
        """Validate license plate format"""
        # Basic Indian format validation
        if len(plate_text) >= 8:
            return bool(self.plate_pattern.match(plate_text.upper()))
        return False
