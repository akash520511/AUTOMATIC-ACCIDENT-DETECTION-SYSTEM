
import cv2
import asyncio
import tempfile
import os
from typing import Tuple, Optional, Generator
import logging

from app.config import Config
from app.logger import logger

class VideoStream:
    """Handle video file and webcam streams"""
    
    def __init__(self):
        self.cap = None
        self.source_type = None
        self.fps = 0
        self.frame_count = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
    
    def open_webcam(self, camera_id: int = 0) -> bool:
        """Open webcam stream"""
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open webcam {camera_id}")
            return False
        
        self.source_type = 'webcam'
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Webcam opened: {self.width}x{self.height} @ {self.fps:.1f}fps")
        return True
    
    def open_video(self, video_path: str) -> bool:
        """Open video file"""
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return False
        
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False
        
        self.source_type = 'video'
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        logger.info(f"Video loaded: {self.width}x{self.height}, {self.fps:.1f}fps, {duration:.1f}s")
        return True
    
    def read_frame(self) -> Tuple[Optional[np.ndarray], bool]:
        """Read next frame"""
        if self.cap is None:
            return None, False
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            return frame, True
        
        # Video ended
        return None, False
    
    def get_progress(self) -> float:
        """Get playback progress"""
        if self.source_type == 'video' and self.total_frames > 0:
            return (self.frame_count / self.total_frames) * 100
        return 0
    
    def restart(self):
        """Restart video"""
        if self.source_type == 'video' and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            logger.info("Video restarted")
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
            logger.info("Video stream released")
