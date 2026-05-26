"""
Point calibrator for side-view knee angle analysis.
Allows user to select 3 tracking points: hip, knee, ankle.
"""

import cv2
from typing import List, Optional, Tuple


class PointCalibrator:
    """Interactive point selection using crosshairs."""
    
    def __init__(self, title: str, num_points: int = 3, point_names: List[str] = None):
        self.title = title
        self.num_points = num_points
        self.point_names = point_names or [f"Point {i+1}" for i in range(num_points)]
        self.points = []
        
    def calibrate(self, frame) -> Optional[List[Tuple[int, int]]]:
        """
        Interactive point selection. Returns list of (x, y) tuples.
        Returns None if cancelled.
        """
        h, w = frame.shape[:2]
        display = frame.copy()
        crosshair_pos = [w // 2, h // 2]
        
        def mouse_callback(event, x, y, flags, param):
            crosshair_pos[0] = x
            crosshair_pos[1] = y
            
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                if len(self.points) < self.num_points:
                    print(f"✓ {self.point_names[len(self.points)-1]} set at ({x}, {y})")
                    print(f"  Click to set {self.point_names[len(self.points)]}")
        
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.title, 1280, 720)
        cv2.setMouseCallback(self.title, mouse_callback)
        
        print("\n" + "=" * 60)
        print(f"🎯 {self.title}")
        print("=" * 60)
        print(f"Instructions:")
        print(f"  1. Move mouse to position, click to select")
        print(f"  2. Select {self.num_points} points in order: {', '.join(self.point_names)}")
        print(f"  3. Press SPACE to confirm")
        print(f"  4. Press ESC to cancel")
        print("=" * 60)
        print(f"Click to set {self.point_names[0]}")
        
        while len(self.points) < self.num_points:
            display = frame.copy()
            
            # Draw existing points
            for i, (px, py) in enumerate(self.points):
                cv2.circle(display, (px, py), 12, (0, 255, 0), -1)
                cv2.circle(display, (px, py), 12, (255, 255, 255), 2)
                cv2.putText(display, self.point_names[i], (px + 15, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw crosshair for next point
            cx, cy = crosshair_pos
            color = (0, 165, 255)  # Orange
            cv2.line(display, (cx - 20, cy), (cx + 20, cy), color, 2)
            cv2.line(display, (cx, cy - 20), (cx, cy + 20), color, 2)
            cv2.circle(display, (cx, cy), 8, color, 1)
            
            # Instructions overlay
            cv2.putText(display, "Select body points", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            progress = f"{len(self.points)}/{self.num_points}"
            cv2.putText(display, progress, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.imshow(self.title, display)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                print("✗ Calibration cancelled")
                cv2.destroyWindow(self.title)
                return None
        
        # All points selected, wait for confirmation
        display = frame.copy()
        for i, (px, py) in enumerate(self.points):
            cv2.circle(display, (px, py), 12, (0, 255, 0), -1)
            cv2.circle(display, (px, py), 12, (255, 255, 255), 2)
            cv2.putText(display, self.point_names[i], (px + 15, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(display, "All points selected! Press SPACE to confirm or ESC to restart",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(self.title, display)
        
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 32:  # SPACE
                print(f"✓ Points confirmed: {self.points}")
                cv2.destroyWindow(self.title)
                return self.points
            elif key == 27:  # ESC
                self.points = []
                print("✗ Restarting calibration...")
                cv2.destroyWindow(self.title)
                return self.calibrate(frame)
