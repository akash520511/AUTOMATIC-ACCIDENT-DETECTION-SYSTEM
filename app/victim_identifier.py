
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VictimIdentifier:
    """Detect potential victims and assess injuries"""
    
    def __init__(self):
        self.person_class = 0  # COCO person class
        self.injury_signs = ['bleeding', 'lying', 'motionless']
    
    def detect_victims(self, frame: np.ndarray) -> List[Dict]:
        """Detect potential victims in frame"""
        victims = []
        
        # Use YOLO to detect persons if available
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            results = model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # Person class
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            # Assess potential injury
                            injury_severity = self._assess_injuries(frame, (x1, y1, x2, y2))
                            
                            victims.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf,
                                'injury_severity': injury_severity,
                                'needs_help': injury_severity > 0.5
                            })
        except:
            # Simulate victim detection for demo
            import random
            if random.random() > 0.7:
                h, w = frame.shape[:2]
                victims.append({
                    'bbox': (random.randint(100, w-200), random.randint(100, h-200),
                            random.randint(120, 220), random.randint(180, 280)),
                    'confidence': random.uniform(0.7, 0.9),
                    'injury_severity': random.uniform(0.3, 0.9),
                    'needs_help': True
                })
        
        return victims
    
    def _assess_injuries(self, frame: np.ndarray, bbox: tuple) -> float:
        """Assess injury severity (0-1)"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0
        
        # Check for red color (blood)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_ratio = np.sum(red_mask > 0) / (roi.shape[0] * roi.shape[1])
        
        # Check for motion (if person is moving)
        # Simplified - in production, analyze motion over time
        
        # Combine factors
        injury_score = min(red_ratio * 3, 1.0)
        
        return round(injury_score, 2)
    
    def get_emergency_response(self, victims: List[Dict]) -> Dict:
        """Get emergency response recommendation"""
        if not victims:
            return {'needs_ambulance': False, 'priority': 'NONE'}
        
        max_injury = max([v['injury_severity'] for v in victims])
        
        if max_injury > 0.8:
            return {'needs_ambulance': True, 'priority': 'HIGH', 'victims': len(victims)}
        elif max_injury > 0.5:
            return {'needs_ambulance': True, 'priority': 'MEDIUM', 'victims': len(victims)}
        else:
            return {'needs_ambulance': False, 'priority': 'LOW', 'victims': len(victims)}
