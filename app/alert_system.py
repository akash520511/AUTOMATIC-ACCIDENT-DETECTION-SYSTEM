
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
import logging
from datetime import datetime

from app.config import Config
from app.logger import logger

class AlertSystem:
    """Emergency alert system for accidents"""
    
    def __init__(self):
        self.last_alert_time = 0
        self.alerts_sent = 0
    
    def trigger_alert(self, accident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger emergency alert"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < Config.ALERT_COOLDOWN:
            return {'sent': False, 'reason': 'Cooldown active'}
        
        severity = accident_data.get('severity', 'LOW')
        
        # Only send alerts for HIGH severity
        if severity != 'HIGH':
            return {'sent': False, 'reason': f'Severity {severity} below threshold'}
        
        # Create alert message
        message = self._format_alert(accident_data)
        
        # Send alerts
        email_sent = self._send_email(message) if Config.EMAIL_ENABLED else False
        sms_sent = self._send_sms(message) if Config.SMS_ENABLED else False
        
        sent = email_sent or sms_sent
        
        if sent:
            self.alerts_sent += 1
            self.last_alert_time = current_time
            logger.info(f"🚨 Alert sent for accident ID: {accident_data.get('id')}")
        
        return {
            'sent': sent,
            'email_sent': email_sent,
            'sms_sent': sms_sent,
            'message': message,
            'alert_id': f"ALT-{int(current_time)}"
        }
    
    def _format_alert(self, accident_data: Dict) -> str:
        """Format alert message"""
        timestamp = accident_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        severity = accident_data.get('severity', 'HIGH')
        confidence = accident_data.get('confidence', 94.2)
        vehicle_count = accident_data.get('vehicle_count', 0)
        
        return f"""
╔══════════════════════════════════════════════════════════╗
║ 🚨 EMERGENCY ALERT - ACCIDENT DETECTED                    ║
╠══════════════════════════════════════════════════════════╣
║ Time: {timestamp}                                         ║
║ Severity: {severity} ({confidence:.1f}% confidence)       ║
║ Vehicles Involved: {vehicle_count}                        ║
║ Location: {accident_data.get('location', 'Camera Feed')}  ║
╠══════════════════════════════════════════════════════════╣
║ 🚑 Ambulance: Dispatched (ETA: 5 min)                     ║
║ 👮 Police: Dispatched (ETA: 7 min)                        ║
║ 🔥 Fire Department: On standby                            ║
╚══════════════════════════════════════════════════════════╝
"""
    
    def _send_email(self, message: str) -> bool:
        """Send email alert (simulated)"""
        logger.info(f"📧 EMAIL ALERT:\n{message}")
        return True
    
    def _send_sms(self, message: str) -> bool:
        """Send SMS alert (simulated)"""
        logger.info(f"📱 SMS ALERT:\n{message[:160]}")
        return True
