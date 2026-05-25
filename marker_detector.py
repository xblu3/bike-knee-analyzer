"""
Color marker detection and tracking for knee angle analysis.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List


class ColorMarkerDetector:
    """Detect and track colored markers on a person."""
    
    def __init__(self, marker_color: str = "red", calibration_mode: bool = False):
        """
        Initialize the color marker detector.
        
        Args:
            marker_color: Color of markers ("red", "white", "blue", "green", "yellow")
            calibration_mode: If True, allow user to select marker color from video
        """
        self.marker_color = marker_color.lower()
        self.calibration_mode = calibration_mode
        self.color_ranges = self._get_color_ranges()
        self.detected_markers = []
    
    def _get_color_ranges(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get HSV color ranges for different colors.
        
        Returns:
            Dictionary mapping color names to (lower_hsv, upper_hsv) tuples
        """
        return {
            'red': (
                np.array([0, 100, 100]),
                np.array([10, 255, 255])
            ),
            'white': (
                np.array([0, 0, 200]),
                np.array([180, 30, 255])
            ),
            'blue': (
                np.array([100, 100, 100]),
                np.array([130, 255, 255])
            ),
            'green': (
                np.array([35, 100, 100]),
                np.array([85, 255, 255])
            ),
            'yellow': (
                np.array([20, 100, 100]),
                np.array([30, 255, 255])
            )
        }
    
    def detect_markers(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect colored markers in a frame.
        
        Args:
            frame: BGR image frame
        
        Returns:
            List of (x, y) coordinates of detected markers, sorted by proximity
        """
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color range
        if self.marker_color not in self.color_ranges:
            print(f"Unknown color: {self.marker_color}. Using red.")
            lower, upper = self.color_ranges['red']
        else:
            lower, upper = self.color_ranges[self.marker_color]
        
        # Create mask for the color
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area to avoid noise
            if 50 < area < 5000:
                # Get center of contour
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    markers.append((cx, cy))
        
        # Sort markers by x-coordinate (left to right)
        markers.sort(key=lambda p: p[0])
        self.detected_markers = markers
        
        return markers
    
    def get_marker_positions(self, frame: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
        """
        Get the positions of hip, knee, and ankle markers.
        
        Args:
            frame: BGR image frame
        
        Returns:
            Dictionary with 'hip', 'knee', 'ankle' positions or None if not all found
        """
        markers = self.detect_markers(frame)
        
        if len(markers) < 3:
            return None
        
        # Take first 3 detected markers as hip, knee, ankle (sorted top to bottom)
        # Assume markers are placed top to bottom
        sorted_by_y = sorted(markers[:3], key=lambda p: p[1])
        
        return {
            'hip': sorted_by_y[0],
            'knee': sorted_by_y[1],
            'ankle': sorted_by_y[2]
        }
    
    def draw_markers(self, frame: np.ndarray, markers: List[Tuple[int, int]],
                    positions: Optional[Dict[str, Tuple[int, int]]] = None) -> np.ndarray:
        """
        Draw detected markers on the frame.
        
        Args:
            frame: BGR image frame
            markers: List of marker positions
            positions: Dictionary with named marker positions
        
        Returns:
            Frame with drawn markers
        """
        frame = frame.copy()
        
        # Draw all detected markers
        color_bgr = self._get_marker_color_bgr()
        
        for i, (x, y) in enumerate(markers):
            cv2.circle(frame, (x, y), 8, color_bgr, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
        
        # Draw labeled positions if provided
        if positions:
            labels = ['hip', 'knee', 'ankle']
            label_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: blue, green, red
            
            for label, color in zip(labels, label_colors):
                if label in positions:
                    x, y = positions[label]
                    cv2.circle(frame, (x, y), 10, color, 2)
                    cv2.putText(frame, label, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, color, 2)
        
        return frame
    
    def _get_marker_color_bgr(self) -> Tuple[int, int, int]:
        """Get BGR color for drawing based on marker color."""
        color_map = {
            'red': (0, 0, 255),
            'white': (255, 255, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255)
        }
        return color_map.get(self.marker_color, (0, 0, 255))
    
    def calibrate_color(self, frame: np.ndarray) -> Optional[str]:
        """
        Allow user to calibrate marker color by clicking on a marker in the frame.
        
        Args:
            frame: BGR image frame for calibration
        
        Returns:
            Selected color or None
        """
        print("\n🎯 Color Calibration Mode")
        print("Click on a marker to detect its color")
        print("Press 'ESC' to skip calibration")
        
        selected_color = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_color
            if event == cv2.EVENT_LBUTTONDOWN:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, s, v = hsv[y, x]
                print(f"Clicked HSV: H={h}, S={s}, V={v}")
                
                # Determine closest color
                if s < 30:
                    selected_color = 'white'
                elif h < 10 or h > 170:
                    selected_color = 'red'
                elif 100 <= h <= 130:
                    selected_color = 'blue'
                elif 35 <= h <= 85:
                    selected_color = 'green'
                elif 20 <= h <= 30:
                    selected_color = 'yellow'
                else:
                    selected_color = 'red'
                
                print(f"✓ Detected color: {selected_color}")
        
        cv2.imshow('Calibration - Click on a marker', frame)
        cv2.setMouseCallback('Calibration - Click on a marker', mouse_callback)
        
        while True:
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break
            elif selected_color:
                break
        
        cv2.destroyWindow('Calibration - Click on a marker')
        return selected_color
