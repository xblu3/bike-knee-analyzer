"""
Multi-angle analyzer for side-view bike riding analysis.
Tracks and displays angles for: shoulders, elbows, hips, knees, ankles.
Similar to the visual analysis shown in the reference image.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import math


def calculate_angle(p1: Tuple[int, int], vertex: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Calculate angle at vertex formed by p1-vertex-p2.
    Returns angle in degrees (0-180).
    """
    x1, y1 = p1
    vx, vy = vertex
    x2, y2 = p2
    
    # Vectors from vertex to p1 and p2
    v1 = np.array([x1 - vx, y1 - vy])
    v2 = np.array([x2 - vx, y2 - vy])
    
    # Angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


class MultiAngleAnalyzer:
    """
    Analyze side-view bike riding with multiple joint angles.
    Displays shoulder, elbow, hip, knee, ankle angles.
    """
    
    # MediaPipe pose landmark indices
    LANDMARKS = {
        'nose': 0,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }
    
    def __init__(self, side: str = 'right'):
        """
        Args:
            side: 'left' or 'right' - which side of body to analyze
        """
        self.side = side
        self.angles_history = {
            'shoulder': [],
            'elbow': [],
            'hip': [],
            'knee': [],
            'ankle': [],
        }
    
    def get_side_landmarks(self, landmarks, h: int, w: int) -> Dict[str, Tuple[int, int]]:
        """Extract landmarks for the specified side."""
        lm_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
        
        if self.side == 'right':
            return {
                'shoulder': self._lm_to_pixel(lm_list[12], h, w),  # right shoulder
                'elbow': self._lm_to_pixel(lm_list[14], h, w),     # right elbow
                'wrist': self._lm_to_pixel(lm_list[16], h, w),     # right wrist
                'hip': self._lm_to_pixel(lm_list[24], h, w),       # right hip
                'knee': self._lm_to_pixel(lm_list[26], h, w),      # right knee
                'ankle': self._lm_to_pixel(lm_list[28], h, w),     # right ankle
            }
        else:  # left
            return {
                'shoulder': self._lm_to_pixel(lm_list[11], h, w),  # left shoulder
                'elbow': self._lm_to_pixel(lm_list[13], h, w),     # left elbow
                'wrist': self._lm_to_pixel(lm_list[15], h, w),     # left wrist
                'hip': self._lm_to_pixel(lm_list[23], h, w),       # left hip
                'knee': self._lm_to_pixel(lm_list[25], h, w),      # left knee
                'ankle': self._lm_to_pixel(lm_list[27], h, w),     # left ankle
            }
    
    def _lm_to_pixel(self, landmark, h: int, w: int) -> Tuple[int, int]:
        """Convert normalized landmark to pixel coordinates."""
        if isinstance(landmark, dict):
            x, y = landmark['x'], landmark['y']
        else:
            x, y = landmark.x, landmark.y
        return (int(x * w), int(y * h))
    
    def analyze_frame(self, landmarks, h: int, w: int) -> Optional[Dict]:
        """
        Analyze frame and calculate all joint angles.
        Returns dict with angle data.
        """
        points = self.get_side_landmarks(landmarks, h, w)
        
        angles = {}
        
        # Shoulder angle: neck-shoulder-elbow
        # (using hip as proxy for neck position on side view)
        angles['shoulder'] = calculate_angle(
            points['hip'],
            points['shoulder'],
            points['elbow']
        )
        
        # Elbow angle: shoulder-elbow-wrist
        angles['elbow'] = calculate_angle(
            points['shoulder'],
            points['elbow'],
            points['wrist']
        )
        
        # Hip angle: shoulder-hip-knee
        angles['hip'] = calculate_angle(
            points['shoulder'],
            points['hip'],
            points['knee']
        )
        
        # Knee angle: hip-knee-ankle
        angles['knee'] = calculate_angle(
            points['hip'],
            points['knee'],
            points['ankle']
        )
        
        # Ankle angle: knee-ankle-ground reference
        # (simplified: use horizontal line as ground)
        ground_ref = (points['ankle'][0] + 50, points['ankle'][1])
        angles['ankle'] = calculate_angle(
            points['knee'],
            points['ankle'],
            ground_ref
        )
        
        # Store history
        for joint, angle in angles.items():
            self.angles_history[joint].append(angle)
        
        return {
            'points': points,
            'angles': angles,
        }
    
    def draw_overlay(self, frame: np.ndarray, data: Dict) -> np.ndarray:
        """Draw skeleton with angle annotations."""
        out = frame.copy()
        points = data['points']
        angles = data['angles']
        
        # Draw skeleton lines and joints
        joints_order = [
            ('shoulder', 'elbow'),
            ('elbow', 'wrist'),
            ('shoulder', 'hip'),
            ('hip', 'knee'),
            ('knee', 'ankle'),
        ]
        
        # Line color
        line_color = (0, 165, 255)  # Orange
        
        for joint1, joint2 in joints_order:
            p1 = points[joint1]
            p2 = points[joint2]
            cv2.line(out, p1, p2, line_color, 3)
        
        # Draw joint circles
        joint_colors = {
            'shoulder': (0, 255, 0),   # Green
            'elbow': (255, 0, 0),      # Blue
            'wrist': (255, 0, 255),    # Magenta
            'hip': (255, 255, 0),      # Cyan
            'knee': (0, 255, 255),     # Yellow
            'ankle': (255, 0, 0),      # Blue
        }
        
        for joint, color in joint_colors.items():
            if joint in points:
                cv2.circle(out, points[joint], 8, color, -1)
                cv2.circle(out, points[joint], 8, (255, 255, 255), 2)
        
        # Draw angles with labels
        angle_positions = {
            'shoulder': (points['shoulder'], 'Shoulder'),
            'elbow': (points['elbow'], 'Elbow'),
            'hip': (points['hip'], 'Hip'),
            'knee': (points['knee'], 'Knee'),
            'ankle': (points['ankle'], 'Ankle'),
        }
        
        for joint, (pos, label) in angle_positions.items():
            if joint in angles:
                angle = angles[joint]
                x, y = pos
                
                # Draw angle value
                text = f"{angle:.1f}°"
                # Color code by angle range
                if 60 <= angle <= 120:
                    text_color = (0, 255, 0)  # Green - good range
                else:
                    text_color = (0, 165, 255)  # Orange - outside range
                
                cv2.putText(out, text, (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(out, label, (x + 15, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Summary box
        cv2.rectangle(out, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.rectangle(out, (10, 10), (400, 160), (0, 165, 255), 2)
        
        y_offset = 35
        cv2.putText(out, f"Side View Analysis ({self.side.upper()})", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        y_offset += 30
        for joint in ['shoulder', 'elbow', 'hip', 'knee', 'ankle']:
            if joint in angles:
                angle = angles[joint]
                cv2.putText(out, f"{joint.capitalize():10s}: {angle:6.1f}°", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 25
        
        return out
    
    def get_statistics(self) -> Dict:
        """Get angle statistics over analyzed frames."""
        stats = {}
        for joint, angles in self.angles_history.items():
            if angles:
                angles_arr = np.array(angles)
                stats[joint] = {
                    'min': float(np.min(angles_arr)),
                    'max': float(np.max(angles_arr)),
                    'avg': float(np.mean(angles_arr)),
                    'std': float(np.std(angles_arr)),
                }
            else:
                stats[joint] = {'min': 0, 'max': 0, 'avg': 0, 'std': 0}
        return stats
    
    def reset(self):
        """Reset history for new analysis."""
        for joint in self.angles_history:
            self.angles_history[joint] = []
