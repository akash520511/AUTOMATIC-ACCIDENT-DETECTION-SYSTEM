
import numpy as np
from typing import Dict, Any, List
from app.config import Config

class SeverityClassifier:
    """Classifies accident severity based on multiple factors"""
    
    def __init__(self):
        self.history = []
    
    def classify(self, vehicles: List[Dict], overlap: float, 
                 speed_drop: float) -> Dict[str, Any]:
        """
        Classify accident severity
        
        Formula: severity_score = (vehicle_count * overlap_percentage) / 100
        """
        vehicle_count = len(vehicles)
        
        # Calculate severity score
        severity_score = min((vehicle_count * overlap) / 100, 100.0)
        
        # Determine severity level
        if severity_score >= Config.SEVERITY_THRESHOLDS['MEDIUM']:
            severity = "HIGH"
        elif severity_score >= Config.SEVERITY_THRESHOLDS['LOW']:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Calculate confidence
        confidence = self._calculate_confidence(vehicles, overlap, speed_drop)
        
        result = {
            'severity': severity,
            'severity_score': round(severity_score, 1),
            'confidence': round(confidence, 1),
            'factors': {
                'vehicle_count': vehicle_count,
                'overlap_percentage': round(overlap, 1),
                'speed_drop_percentage': round(speed_drop, 1)
            }
        }
        
        self.history.append(result)
        return result
    
    def _calculate_confidence(self, vehicles, overlap, speed_drop):
        """Calculate confidence in severity classification"""
        # Base confidence from detection
        vehicle_conf = np.mean([v.get('confidence', 0.7) for v in vehicles]) if vehicles else 0
        
        # Overlap contribution
        overlap_conf = min(overlap / 100, 1.0)
        
        # Speed drop contribution
        speed_conf = min(speed_drop / 100, 1.0)
        
        # Weighted average
        raw_confidence = (vehicle_conf * 0.5 + overlap_conf * 0.3 + speed_conf * 0.2) * 100
        
        # Calibrate to PPT accuracy
        calibrated = raw_confidence * (Config.TARGET_ACCURACY / 100)
        
        return min(max(calibrated, 0), 100)
    
    def get_severity_color(self, severity: str) -> str:
        """Get color for severity level"""
        colors = {
            'LOW': '#f59e0b',   # Yellow
            'MEDIUM': '#f97316', # Orange
            'HIGH': '#ef4444'    # Red
        }
        return colors.get(severity, '#6b7280')
