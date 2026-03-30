
import logging
from typing import Dict, Any, List
from datetime import datetime

from app.database import db
from app.logger import logger

class FamilyNotifier:
    """Notify family members about involved vehicles"""
    
    def __init__(self):
        self.family_contacts = {}
        self._load_contacts()
    
    def _load_contacts(self):
        """Load family contacts from database"""
        # In production, fetch from database
        self.family_contacts = {
            "KA01AB1234": {"name": "Ramesh Kumar", "phone": "9876543210", "email": "ramesh@example.com"},
            "MH02CD5678": {"name": "Suresh Reddy", "phone": "9876543211", "email": "suresh@example.com"},
            "DL03EF9012": {"name": "Priya Sharma", "phone": "9876543212", "email": "priya@example.com"},
            "TN04GH3456": {"name": "Arun Kumar", "phone": "9876543213", "email": "arun@example.com"},
        }
    
    def notify_family(self, plate_numbers: List[str], accident_data: Dict) -> List[Dict]:
        """Notify family members of involved vehicles"""
        notifications = []
        
        for plate in plate_numbers:
            if plate in self.family_contacts:
                contact = self.family_contacts[plate]
                message = self._create_message(plate, contact, accident_data)
                
                # Send notification
                sent = self._send_notification(contact, message)
                
                notifications.append({
                    'plate_number': plate,
                    'contact_name': contact['name'],
                    'sent': sent,
                    'message': message
                })
                
                logger.info(f"Family notified for plate: {plate}")
        
        return notifications
    
    def _create_message(self, plate: str, contact: Dict, accident_data: Dict) -> str:
        """Create notification message"""
        timestamp = accident_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        severity = accident_data.get('severity', 'UNKNOWN')
        location = accident_data.get('location', 'Unknown location')
        
        return f"""
Dear {contact['name']},

This is an automated alert from the Accident Detection System.

Vehicle {plate} was involved in a {severity} accident at {timestamp}.
Location: {location}

Emergency services have been dispatched to the location.
Please contact emergency services for more information.

- Accident Detection System
"""
    
    def _send_notification(self, contact: Dict, message: str) -> bool:
        """Send notification (simulated)"""
        logger.info(f"📱 NOTIFYING {contact['name']} ({contact['phone']}):\n{message[:200]}...")
        return True
    
    def get_vehicle_owner(self, plate_number: str) -> Dict:
        """Get owner information for a vehicle"""
        return self.family_contacts.get(plate_number, None)
