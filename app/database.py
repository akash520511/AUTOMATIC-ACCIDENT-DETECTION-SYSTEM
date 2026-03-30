
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.config import Config
from app.logger import logger

class Database:
    """SQLite database handler for accident records"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or Config.DATABASE_PATH
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Accidents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accidents (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    vehicle_count INTEGER NOT NULL,
                    overlap_percentage REAL,
                    speed_drop_percentage REAL,
                    detection_time_ms REAL,
                    location TEXT,
                    video_path TEXT,
                    snapshot_path TEXT,
                    heatmap_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    accident_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    recipient TEXT,
                    message TEXT,
                    status TEXT DEFAULT 'PENDING',
                    sent_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (accident_id) REFERENCES accidents(id)
                )
            ''')
            
            # Number plates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS number_plates (
                    id TEXT PRIMARY KEY,
                    accident_id TEXT,
                    plate_number TEXT NOT NULL,
                    confidence REAL,
                    snapshot_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (accident_id) REFERENCES accidents(id)
                )
            ''')
            
            # Victims table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS victims (
                    id TEXT PRIMARY KEY,
                    accident_id TEXT,
                    plate_number TEXT,
                    estimated_injuries TEXT,
                    snapshot_path TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (accident_id) REFERENCES accidents(id)
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    avg_response_time_ms REAL,
                    total_accidents INTEGER,
                    total_alerts INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def save_accident(self, accident_data: Dict[str, Any]) -> str:
        """Save accident record"""
        import uuid
        
        accident_id = str(uuid.uuid4())[:8]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO accidents (
                    id, timestamp, severity, confidence, vehicle_count,
                    overlap_percentage, speed_drop_percentage, detection_time_ms,
                    location, video_path, snapshot_path, heatmap_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                accident_id,
                accident_data.get('timestamp', datetime.now().isoformat()),
                accident_data.get('severity', 'LOW'),
                accident_data.get('confidence', 0),
                accident_data.get('vehicle_count', 0),
                accident_data.get('overlap_percentage', 0),
                accident_data.get('speed_drop_percentage', 0),
                accident_data.get('detection_time_ms', 0),
                accident_data.get('location', ''),
                accident_data.get('video_path', ''),
                accident_data.get('snapshot_path', ''),
                json.dumps(accident_data.get('heatmap_grid', []))
            ))
            conn.commit()
        
        logger.info(f"Accident saved: {accident_id} - {accident_data.get('severity')}")
        return accident_id
    
    def save_alert(self, alert_data: Dict[str, Any]) -> str:
        """Save alert record"""
        import uuid
        
        alert_id = str(uuid.uuid4())[:8]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (
                    id, accident_id, alert_type, severity, recipient, message, status, sent_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_id,
                alert_data.get('accident_id'),
                alert_data.get('alert_type', 'EMAIL'),
                alert_data.get('severity'),
                alert_data.get('recipient'),
                alert_data.get('message'),
                alert_data.get('status', 'PENDING'),
                alert_data.get('sent_at')
            ))
            conn.commit()
        
        return alert_id
    
    def save_number_plate(self, plate_data: Dict[str, Any]) -> str:
        """Save detected number plate"""
        import uuid
        
        plate_id = str(uuid.uuid4())[:8]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO number_plates (id, accident_id, plate_number, confidence, snapshot_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                plate_id,
                plate_data.get('accident_id'),
                plate_data.get('plate_number'),
                plate_data.get('confidence'),
                plate_data.get('snapshot_path')
            ))
            conn.commit()
        
        return plate_id
    
    def get_recent_accidents(self, limit: int = 10) -> List[Dict]:
        """Get recent accidents"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM accidents 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM alerts 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total accidents
            cursor.execute("SELECT COUNT(*) FROM accidents")
            total_accidents = cursor.fetchone()[0]
            
            # Accidents by severity
            cursor.execute("""
                SELECT severity, COUNT(*) 
                FROM accidents 
                GROUP BY severity
            """)
            severity_counts = dict(cursor.fetchall())
            
            # Total alerts
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]
            
            return {
                'total_accidents': total_accidents,
                'severity_counts': severity_counts,
                'total_alerts': total_alerts
            }
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO metrics (
                    date, accuracy, precision, recall, f1_score,
                    avg_response_time_ms, total_accidents, total_alerts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime("%Y-%m-%d"),
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                metrics.get('avg_response_time_ms', 0),
                metrics.get('total_accidents', 0),
                metrics.get('total_alerts', 0)
            ))
            conn.commit()

# Global database instance
db = Database()
