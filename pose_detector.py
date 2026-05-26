"""
Pose detection using MediaPipe.
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple

try:
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    Pose = solutions.pose.Pose
    PoseLandmark = solutions.pose.PoseLandmark
except ImportError as e:
    raise ImportError(f"MediaPipe not properly installed. Run: pip install --upgrade mediapipe\nError: {e}")


class PoseDetector:
    """Wrapper around MediaPipe Pose for detecting body landmarks."""
    
    # Landmark indices
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    
    def __init__(self, static_image_mode: bool = False, 
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the pose detector.
        
        Args:
            static_image_mode: Whether to treat input as static image or video stream
            model_complexity: 0 (lite), 1 (full), 2 (heavy)
            smooth_landmarks: Whether to smooth landmarks across frames
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        try:
            self.pose = Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Pose: {e}")
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect pose landmarks in an image.
        
        Args:
            image: BGR image from OpenCV
        
        Returns:
            Dictionary with landmarks or None if no pose detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extract landmarks
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return landmarks
    
    def get_landmark_coords(self, landmarks: Dict, idx: int, 
                           image_width: int, image_height: int) -> Tuple[float, float]:
        """
        Get pixel coordinates for a landmark.
        
        Args:
            landmarks: Landmarks dictionary from detect()
            idx: Landmark index
            image_width: Width of image in pixels
            image_height: Height of image in pixels
        
        Returns:
            (x, y) tuple in pixel coordinates
        """
        if idx not in landmarks:
            return (0, 0)
        
        lm = landmarks[idx]
        x = lm['x'] * image_width
        y = lm['y'] * image_height
        
        return (x, y)
    
    def get_knee_landmarks(self, landmarks: Dict, image_width: int, 
                          image_height: int, side: str = 'right') -> Optional[Dict]:
        """
        Extract knee joint landmarks (hip, knee, ankle).
        
        Args:
            landmarks: Landmarks dictionary from detect()
            image_width: Width of image in pixels
            image_height: Height of image in pixels
            side: 'left' or 'right'
        
        Returns:
            Dictionary with 'hip', 'knee', 'ankle' coordinates or None
        """
        if side == 'right':
            hip_idx = self.RIGHT_HIP
            knee_idx = self.RIGHT_KNEE
            ankle_idx = self.RIGHT_ANKLE
        else:
            hip_idx = self.LEFT_HIP
            knee_idx = self.LEFT_KNEE
            ankle_idx = self.LEFT_ANKLE
        
        # Check if all landmarks are visible
        if (hip_idx not in landmarks or knee_idx not in landmarks or 
            ankle_idx not in landmarks):
            return None
        
        hip_vis = landmarks[hip_idx]['visibility']
        knee_vis = landmarks[knee_idx]['visibility']
        ankle_vis = landmarks[ankle_idx]['visibility']
        
        # Require minimum visibility confidence
        if hip_vis < 0.5 or knee_vis < 0.5 or ankle_vis < 0.5:
            return None
        
        return {
            'hip': self.get_landmark_coords(landmarks, hip_idx, image_width, image_height),
            'knee': self.get_landmark_coords(landmarks, knee_idx, image_width, image_height),
            'ankle': self.get_landmark_coords(landmarks, ankle_idx, image_width, image_height),
            'hip_visibility': hip_vis,
            'knee_visibility': knee_vis,
            'ankle_visibility': ankle_vis
        }
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
