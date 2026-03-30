
import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple, Optional
import logging

from app.config import Config
from app.logger import logger

class UploadHandler:
    """Handle video file uploads"""
    
    ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    
    def __init__(self):
        self.upload_folder = Config.UPLOAD_FOLDER
    
    def save_upload(self, file) -> Tuple[bool, Optional[str], Optional[str]]:
        """Save uploaded video file"""
        # Check extension
        ext = Path(file.filename).suffix.lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, None, f"Invalid file type. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}"
        
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = self.upload_folder / filename
        
        try:
            # Save file
            with open(filepath, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            
            file_size = filepath.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                filepath.unlink()
                return False, None, f"File too large. Max size: {self.MAX_FILE_SIZE / (1024*1024):.0f}MB"
            
            logger.info(f"File uploaded: {filename} ({file_size / (1024*1024):.1f}MB)")
            return True, str(filepath), None
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False, None, str(e)
    
    def get_uploaded_videos(self) -> list:
        """Get list of uploaded videos"""
        videos = []
        for filepath in self.upload_folder.iterdir():
            if filepath.suffix.lower() in self.ALLOWED_EXTENSIONS:
                stat = filepath.stat()
                videos.append({
                    'filename': filepath.name,
                    'path': str(filepath),
                    'size': stat.st_size,
                    'size_mb': round(stat.st_size / (1024*1024), 2),
                    'created': stat.st_ctime
                })
        
        return sorted(videos, key=lambda x: x['created'], reverse=True)
    
    def delete_video(self, filename: str) -> bool:
        """Delete uploaded video"""
        filepath = self.upload_folder / filename
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Video deleted: {filename}")
            return True
        return False
