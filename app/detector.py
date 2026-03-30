
import cv2
import numpy as np
import asyncio
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLOv8 based vehicle detector"""
    
    def __init__(self):
        self.model = None
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        self.loaded = False
        
    async def load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            from app.config import Config
            self.model = YOLO(Config.YOLO_MODEL)
            self.loaded = True
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.warning(f"YOLO not available: {e} - using simulation mode")
            self.loaded = True  # Fallback to simulation
        return True
    
    async def detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vehicles in frame"""
        if not self.loaded:
            await self.load_model()
        
        if self.model:
            # Use real YOLO
            loop = asyncio.get_event_loop()
            
            def _detect():
                results = self.model(frame, verbose=False)
                vehicles = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls = int(box.cls[0])
                            if cls in self.vehicle_classes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0])
                                vehicles.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': conf,
                                    'class': cls,
                                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                    'area': (x2 - x1) * (y2 - y1)
                                })
                return vehicles
            
            return await loop.run_in_executor(None, _detect)
        else:
            # Simulate vehicle detection for demo
            return self._simulate_vehicles(frame)
    
    def _simulate_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Simulate vehicle detection for demo"""
        import random
        h, w = frame.shape[:2]
        vehicles = []
        
        # Add 1-4 simulated vehicles
        num_vehicles = random.randint(1, 4)
        for i in range(num_vehicles):
            x = random.randint(100, w-200)
            y = random.randint(100, h-200)
            width = random.randint(80, 150)
            height = random.randint(80, 150)
            
            vehicles.append({
                'bbox': (x, y, x+width, y+height),
                'confidence': random.uniform(0.7, 0.95),
                'center': (x+width//2, y+height//2),
                'area': width * height
            })
        
        return vehicles
    
    def calculate_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between bounding boxes"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        # Intersection
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right > x_left and y_bottom > y_top:
            intersection = (x_right - x_left) * (y_bottom - y_top)
        else:
            return 0.0
        
        # Union
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return (intersection / union) * 100 if union > 0 else 0.0
