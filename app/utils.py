
import cv2
import numpy as np
import base64
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging

from app.config import Config
from app.logger import logger

class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def base64_to_cv2(image_b64: str) -> np.ndarray:
        """Convert base64 image to OpenCV format"""
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    
    @staticmethod
    def cv2_to_base64(frame: np.ndarray) -> str:
        """Convert OpenCV image to base64"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def resize_with_aspect(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        if width is None:
            ratio = height / h
            dim = (int(w * ratio), height)
        else:
            ratio = width / w
            dim = (width, int(h * ratio))
        
        return cv2.resize(image, dim)
    
    @staticmethod
    def draw_bbox(frame: np.ndarray, bbox: Tuple, color: Tuple, label: str = None) -> np.ndarray:
        """Draw bounding box with label"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if label:
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame

class MetricsCalculator:
    """Calculate performance metrics"""
    
    @staticmethod
    def get_confusion_matrix() -> Dict:
        """Get confusion matrix from PPT"""
        return {
            'true_positives': Config.TRUE_POSITIVES,
            'true_negatives': Config.TRUE_NEGATIVES,
            'false_positives': Config.FALSE_POSITIVES,
            'false_negatives': Config.FALSE_NEGATIVES
        }
    
    @staticmethod
    def calculate_metrics() -> Dict:
        """Calculate metrics from confusion matrix"""
        cm = MetricsCalculator.get_confusion_matrix()
        tp = cm['true_positives']
        tn = cm['true_negatives']
        fp = cm['false_positives']
        fn = cm['false_negatives']
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total * 100
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': round(accuracy, 1),
            'precision': round(precision, 1),
            'recall': round(recall, 1),
            'f1_score': round(f1, 1),
            'total_samples': total
        }
    
    @staticmethod
    def get_response_time() -> float:
        """Get response time from PPT"""
        return Config.TARGET_RESPONSE_TIME

class HeatmapGenerator:
    """Generate collision heatmaps"""
    
    def __init__(self):
        self.impact_points = []
        self.resolution = Config.HEATMAP_RESOLUTION if hasattr(Config, 'HEATMAP_RESOLUTION') else 64
    
    def generate(self, frame_shape: Tuple, vehicles: List[Dict], severity_score: float) -> List[List[float]]:
        """Generate heatmap grid"""
        h, w = frame_shape[:2]
        grid_h = self.resolution
        grid_w = self.resolution
        
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        scale_x = grid_w / w
        scale_y = grid_h / h
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            gx = int(cx * scale_x)
            gy = int(cy * scale_y)
            
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                intensity = severity_score / 100 * vehicle.get('confidence', 0.5)
                self._add_heat_spot(heatmap, gx, gy, 5, intensity)
        
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap.tolist()
    
    def _add_heat_spot(self, heatmap: np.ndarray, cx: int, cy: int, radius: int, intensity: float):
        """Add Gaussian heat spot"""
        h, w = heatmap.shape
        for y in range(max(0, cy - radius), min(h, cy + radius)):
            for x in range(max(0, cx - radius), min(w, cx + radius)):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius:
                    heatmap[y, x] += intensity * (1 - dist / radius)
