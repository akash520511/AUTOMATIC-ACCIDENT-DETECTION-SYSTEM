
import sys
import os
import uvicorn
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import Config
from app.logger import logger

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Automatic Accident Detection System")
    parser.add_argument("--host", default=Config.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=Config.API_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=Config.DEBUG, help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚗 AUTOMATIC ACCIDENT DETECTION SYSTEM")
    print("="*70)
    print("Team: Akash R (24MIS0132), Boopathi R (24MIS0499), Vignesh E (24MIS0559)")
    print("="*70)
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📊 Accuracy: {Config.TARGET_ACCURACY}%")
    print(f"⚡ Response Time: {Config.TARGET_RESPONSE_TIME}s")
    print("="*70)
    print("\n📋 Available Features:")
    print("   ✅ Real-time Accident Detection")
    print("   ✅ Severity Classification (LOW/MEDIUM/HIGH)")
    print("   ✅ Number Plate Detection")
    print("   ✅ Victim Identification")
    print("   ✅ Family Notification")
    print("   ✅ Emergency Alerts")
    print("   ✅ Dashboard & Analytics")
    print("="*70)
    print("\n🚀 Starting server... Press Ctrl+C to stop\n")
    
    # Create API app
    from app.api import create_app
    app = create_app()
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
