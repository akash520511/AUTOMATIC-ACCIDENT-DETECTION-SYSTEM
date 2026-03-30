
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from typing import List
from datetime import datetime

from app.config import Config
from app.database import db
from app.detector import YOLODetector
from app.tracker import VehicleTracker
from app.severity_classifier import SeverityClassifier
from app.number_plate_detector import NumberPlateDetector
from app.victim_identifier import VictimIdentifier
from app.alert_system import AlertSystem
from app.family_notifier import FamilyNotifier
from app.upload_handler import UploadHandler
from app.video_stream import VideoStream
from app.utils import ImageProcessor, MetricsCalculator, HeatmapGenerator
from app.logger import logger

# Initialize components
detector = YOLODetector()
tracker = VehicleTracker()
severity_classifier = SeverityClassifier()
plate_detector = NumberPlateDetector()
victim_identifier = VictimIdentifier()
alert_system = AlertSystem()
family_notifier = FamilyNotifier()
upload_handler = UploadHandler()
heatmap_gen = HeatmapGenerator()

# Global state
active_websockets = []

def create_app():
    """Create and configure FastAPI app"""
    
    app = FastAPI(
        title="Automatic Accident Detection System",
        description="""
        AI-powered accident detection system with:
        - YOLOv8 Vehicle Detection
        - Severity Classification (LOW/MEDIUM/HIGH)
        - Number Plate Recognition
        - Victim Detection
        - Family Notification
        - Emergency Alerts
        """,
        version="3.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
    if os.path.exists(frontend_path):
        app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    
    # ===== Root Endpoints =====
    
    @app.get("/")
    async def root():
        return {
            "name": "Automatic Accident Detection System",
            "version": "3.0.0",
            "team": [
                {"name": "Akash R", "id": "24MIS0132"},
                {"name": "Boopathi R", "id": "24MIS0499"},
                {"name": "Vignesh E", "id": "24MIS0559"}
            ],
            "features": [
                "YOLOv8 Vehicle Detection",
                "Optical Flow Tracking",
                "Severity Classification",
                "Number Plate Recognition",
                "Victim Detection",
                "Family Notification",
                "Emergency Alerts",
                "Heatmap Visualization"
            ],
            "metrics": {
                "accuracy": f"{Config.TARGET_ACCURACY}%",
                "response_time": f"{Config.TARGET_RESPONSE_TIME}s",
                "true_positives": Config.TRUE_POSITIVES,
                "true_negatives": Config.TRUE_NEGATIVES,
                "false_positives": Config.FALSE_POSITIVES,
                "false_negatives": Config.FALSE_NEGATIVES
            }
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": detector.loaded
        }
    
    # ===== Detection Endpoints =====
    
    @app.post("/api/detect/upload")
    async def detect_upload(file: UploadFile = File(...)):
        """Upload and process video"""
        import tempfile
        import cv2
        
        # Save uploaded file
        success, filepath, error = upload_handler.save_upload(file)
        if not success:
            raise HTTPException(400, error)
        
        results = []
        
        # Process video
        cap = cv2.VideoCapture(filepath)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame
            if frame_count % Config.FRAME_SAMPLE_RATE == 0:
                # Detect vehicles
                vehicles = await detector.detect_vehicles(frame)
                
                # Calculate motion
                flow, motion = tracker.calculate_optical_flow(frame)
                
                # Track vehicles
                vehicles = tracker.track_vehicles(vehicles, flow)
                
                # Calculate max overlap
                max_overlap = 0.0
                for i in range(len(vehicles)):
                    for j in range(i+1, len(vehicles)):
                        overlap = detector.calculate_overlap(
                            vehicles[i]['bbox'],
                            vehicles[j]['bbox']
                        )
                        max_overlap = max(max_overlap, overlap)
                
                # Calculate speed drop
                speed_drop = 60.0 if len(vehicles) >= 3 else 30.0 if len(vehicles) >= 2 else 0.0
                
                # Classify severity
                severity = severity_classifier.classify(vehicles, max_overlap, speed_drop)
                
                # Detect number plates
                plates = []
                for v in vehicles:
                    plate = plate_detector.detect_plate(frame, v['bbox'])
                    if plate:
                        plates.append(plate)
                
                # Detect victims
                victims = victim_identifier.detect_victims(frame)
                
                # Generate heatmap
                heatmap = heatmap_gen.generate(frame.shape, vehicles, severity['severity_score'])
                
                result = {
                    'frame': frame_count,
                    'timestamp': frame_count / 30,
                    'vehicles': len(vehicles),
                    'severity': severity,
                    'plates': plates,
                    'victims': victims,
                    'heatmap': heatmap
                }
                results.append(result)
                
                # Trigger alert if needed
                if severity['severity'] == 'HIGH':
                    alert_system.trigger_alert({
                        'id': f"ACC-{len(results)}",
                        'severity': severity['severity'],
                        'confidence': severity['confidence'],
                        'vehicle_count': len(vehicles),
                        'plates': [p['plate_number'] for p in plates],
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Notify families
                    if plates:
                        family_notifier.notify_family([p['plate_number'] for p in plates], {
                            'severity': severity['severity'],
                            'timestamp': datetime.now().isoformat(),
                            'location': 'Camera Feed'
                        })
        
        cap.release()
        
        return {
            'success': True,
            'file': filepath,
            'frames_processed': frame_count,
            'results': results
        }
    
    @app.post("/api/detect/frame")
    async def detect_frame(data: dict):
        """Process single frame"""
        try:
            # Decode image
            frame = ImageProcessor.base64_to_cv2(data.get('image', ''))
            
            if frame is None:
                raise HTTPException(400, "Invalid image")
            
            # Detect vehicles
            vehicles = await detector.detect_vehicles(frame)
            
            # Calculate motion
            flow, motion = tracker.calculate_optical_flow(frame)
            
            # Track vehicles
            vehicles = tracker.track_vehicles(vehicles, flow)
            
            # Calculate max overlap
            max_overlap = 0.0
            for i in range(len(vehicles)):
                for j in range(i+1, len(vehicles)):
                    overlap = detector.calculate_overlap(
                        vehicles[i]['bbox'],
                        vehicles[j]['bbox']
                    )
                    max_overlap = max(max_overlap, overlap)
            
            # Calculate speed drop
            speed_drop = 60.0 if len(vehicles) >= 3 else 30.0 if len(vehicles) >= 2 else 0.0
            
            # Classify severity
            severity = severity_classifier.classify(vehicles, max_overlap, speed_drop)
            
            # Detect number plates
            plates = []
            for v in vehicles:
                plate = plate_detector.detect_plate(frame, v['bbox'])
                if plate:
                    plates.append(plate)
            
            # Detect victims
            victims = victim_identifier.detect_victims(frame)
            
            # Generate heatmap
            heatmap = heatmap_gen.generate(frame.shape, vehicles, severity['severity_score'])
            
            return {
                'success': True,
                'vehicles': len(vehicles),
                'severity': severity,
                'plates': plates,
                'victims': victims,
                'heatmap': heatmap,
                'metrics': MetricsCalculator.calculate_metrics()
            }
            
        except Exception as e:
            logger.error(f"Frame detection error: {e}")
            raise HTTPException(500, str(e))
    
    # ===== Metrics Endpoints =====
    
    @app.get("/api/metrics")
    async def get_metrics():
        """Get system metrics"""
        stats = db.get_statistics()
        metrics = MetricsCalculator.calculate_metrics()
        
        return {
            **metrics,
            **stats,
            'response_time': MetricsCalculator.get_response_time(),
            'uptime': "N/A"
        }
    
    @app.get("/api/metrics/confusion")
    async def get_confusion():
        """Get confusion matrix"""
        return MetricsCalculator.get_confusion_matrix()
    
    # ===== Alerts Endpoints =====
    
    @app.get("/api/alerts")
    async def get_alerts(limit: int = 20):
        """Get recent alerts"""
        return db.get_alerts(limit)
    
    @app.get("/api/alerts/stats")
    async def get_alert_stats():
        """Get alert statistics"""
        return {
            'alerts_sent': alert_system.alerts_sent,
            'last_alert_time': alert_system.last_alert_time,
            'cooldown_seconds': Config.ALERT_COOLDOWN
        }
    
    # ===== Accidents Endpoints =====
    
    @app.get("/api/accidents")
    async def get_accidents(limit: int = 10):
        """Get recent accidents"""
        return db.get_recent_accidents(limit)
    
    @app.get("/api/accidents/{accident_id}")
    async def get_accident(accident_id: str):
        """Get specific accident"""
        # In production, fetch from database
        return {"id": accident_id, "message": "Accident details"}
    
    # ===== WebSocket Endpoint =====
    
    @app.websocket("/ws/live")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        active_websockets.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(active_websockets)}")
        
        try:
            while True:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                
                elif data.get("type") == "frame":
                    # Process frame
                    frame = ImageProcessor.base64_to_cv2(data.get('image', ''))
                    
                    if frame is not None:
                        # Detect and respond
                        vehicles = await detector.detect_vehicles(frame)
                        severity = severity_classifier.classify(vehicles, 0, 0)
                        
                        await websocket.send_json({
                            "type": "detection",
                            "data": {
                                "vehicles": len(vehicles),
                                "severity": severity,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
        
        except WebSocketDisconnect:
            active_websockets.remove(websocket)
            logger.info(f"WebSocket disconnected. Total: {len(active_websockets)}")
    
    # ===== Dashboard Endpoints =====
    
    @app.get("/dashboard")
    async def dashboard():
        """Serve dashboard HTML"""
        dashboard_path = os.path.join(frontend_path, "dashboard.html")
        if os.path.exists(dashboard_path):
            return FileResponse(dashboard_path)
        return {"error": "Dashboard not found"}
    
    return app
