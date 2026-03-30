
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class VehicleTracker:
    """Track vehicles across frames using optical flow"""
    
    def __init__(self):
        self.prev_gray = None
        self.prev_vehicles = []
        self.tracks = {}
        self.next_id = 0
        
    def calculate_optical_flow(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate optical flow and return motion magnitude"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0.0
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(mag)
        
        self.prev_gray = gray
        return flow, avg_magnitude
    
    def track_vehicles(self, vehicles: List[Dict], flow: np.ndarray) -> List[Dict]:
        """Track vehicles and update IDs"""
        if not vehicles:
            self.prev_vehicles = []
            return vehicles
        
        if not self.prev_vehicles:
            # First frame - assign new IDs
            for v in vehicles:
                v['track_id'] = self._get_new_id()
            self.prev_vehicles = vehicles
            return vehicles
        
        # Match vehicles using IoU
        for curr in vehicles:
            best_match = None
            best_iou = 0
            
            for prev in self.prev_vehicles:
                iou = self._calculate_iou(curr['bbox'], prev['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = prev
            
            if best_match:
                curr['track_id'] = best_match.get('track_id')
                curr['velocity'] = self._calculate_velocity(
                    curr['center'], best_match['center']
                )
            else:
                curr['track_id'] = self._get_new_id()
                curr['velocity'] = (0, 0)
        
        self.prev_vehicles = vehicles
        return vehicles
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right > x_left and y_bottom > y_top:
            intersection = (x_right - x_left) * (y_bottom - y_top)
        else:
            return 0.0
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _calculate_velocity(self, curr_center, prev_center):
        """Calculate velocity vector"""
        return (
            curr_center[0] - prev_center[0],
            curr_center[1] - prev_center[1]
        )
    
    def _get_new_id(self):
        """Get new tracking ID"""
        self.next_id += 1
        return self.next_id
    
    def reset(self):
        """Reset tracker state"""
        self.prev_gray = None
        self.prev_vehicles = []
        self.tracks = {}
        self.next_id = 0
