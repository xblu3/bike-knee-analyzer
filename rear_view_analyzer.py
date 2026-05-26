"""
Rear-view analyzer for detecting bike body sway using manually calibrated points.
User draws seatpost line, selects left/right shoulder tracking points,
then path is visualized throughout the video.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque


class PointCalibrator:
    """Interactive point selection using crosshairs."""
    
    def __init__(self, title: str, num_points: int = 2, point_names: List[str] = None):
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
            cv2.putText(display, "Select shoulder points", (20, 40),
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


class LineDrawer:
    """Interactive line drawing for seatpost - fixed 200px vertical line."""
    
    def __init__(self, title: str):
        self.title = title
        self.center_x = None
        self.line_length = 200  # Fixed length
        
    def draw_line(self, frame) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Place a fixed-length vertical line. User moves mouse horizontally and clicks to place.
        Returns ((x, y_top), (x, y_bottom)) or None if cancelled.
        """
        h, w = frame.shape[:2]
        crosshair_x = w // 2
        crosshair_y = h // 2
        line_set = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal crosshair_x, line_set
            if event == cv2.EVENT_MOUSEMOVE:
                crosshair_x = x
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.center_x = x
                line_set = True
                print(f"✓ Seatpost line placed at x={x}")
        
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.title, 1280, 720)
        cv2.setMouseCallback(self.title, mouse_callback)
        
        print("\n" + "=" * 60)
        print("🎯 SEATPOST LINE CALIBRATION")
        print("=" * 60)
        print("Instructions:")
        print("  1. Move mouse left/right to position the seatpost line")
        print("  2. LEFT-CLICK to place the line")
        print("  3. Press SPACE to confirm")
        print("  4. Press ESC to cancel")
        print("=" * 60 + "\n")
        
        while True:
            display = frame.copy()
            
            # Draw preview line at crosshair position
            y_top = max(0, crosshair_y - self.line_length // 2)
            y_bottom = min(h - 1, crosshair_y + self.line_length // 2)
            
            if line_set and self.center_x is not None:
                # Draw placed line in bright cyan
                cv2.line(display, (self.center_x, y_top), (self.center_x, y_bottom), (255, 255, 0), 4)
                cv2.circle(display, (self.center_x, y_top), 8, (0, 255, 0), -1)
                cv2.circle(display, (self.center_x, y_bottom), 8, (0, 255, 0), -1)
            else:
                # Draw preview line in orange
                cv2.line(display, (crosshair_x, y_top), (crosshair_x, y_bottom), (0, 165, 255), 3)
                cv2.circle(display, (crosshair_x, crosshair_y), 12, (0, 165, 255), -1)
            
            # Instructions
            cv2.putText(display, "Place Seatpost Line (200px vertical)", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            status = "Line placed" if line_set else "Preview"
            cv2.putText(display, status, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(display, "Move mouse | LEFT-CLICK to place | SPACE=confirm | ESC=cancel", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow(self.title, display)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 32 and self.center_x is not None:  # SPACE
                y_top = max(0, crosshair_y - self.line_length // 2)
                y_bottom = min(h - 1, crosshair_y + self.line_length // 2)
                print(f"✓ Seatpost line confirmed at x={self.center_x}")
                cv2.destroyWindow(self.title)
                return ((self.center_x, y_top), (self.center_x, y_bottom))
            elif key == 27:  # ESC
                print("✗ Calibration cancelled")
                cv2.destroyWindow(self.title)
                return None


class RearViewAnalyzer:
    """
    Analyze bike riding from rear view using manually calibrated points.
    Tracks left/right shoulder points and visualizes their paths.
    """

    def __init__(self, window_size: int = 30, path_length: int = 100):
        """
        Args:
            window_size: Number of frames for smoothing sway detection
            path_length: Number of frames to keep in path history
        """
        self.window_size = window_size
        self.path_length = path_length
        
        # Calibration
        self.left_shoulder_point = None
        self.right_shoulder_point = None
        self.seatpost_line = None  # ((x1, y1), (x2, y2))
        
        # Tracking data
        self.left_path = deque(maxlen=path_length)
        self.right_path = deque(maxlen=path_length)
        self.midpoint_path = deque(maxlen=path_length)
        
        self.marker_xs = deque(maxlen=window_size)
        self.frame_deviations = []
        self.frame_numbers = []
        self.sway_sides = []

    def calibrate_shoulder_points(self, frame) -> bool:
        """Let user select left and right shoulder points."""
        calibrator = PointCalibrator(
            "Shoulder Point Calibration",
            num_points=2,
            point_names=["Left Shoulder", "Right Shoulder"]
        )
        
        points = calibrator.calibrate(frame)
        if points is None:
            return False
        
        self.left_shoulder_point = points[0]
        self.right_shoulder_point = points[1]
        
        print(f"✓ Shoulder points calibrated:")
        print(f"  Left:  {self.left_shoulder_point}")
        print(f"  Right: {self.right_shoulder_point}")
        return True

    def calibrate_seatpost_line(self, frame) -> bool:
        """Let user draw the seatpost line."""
        drawer = LineDrawer("Seatpost Line Calibration")
        line = drawer.draw_line(frame)
        
        if line is None:
            return False
        
        self.seatpost_line = line
        
        # Extract seatpost x position (average of start and end x)
        x1, x2 = line[0][0], line[1][0]
        self.seatpost_x = (x1 + x2) // 2
        
        print(f"✓ Seatpost line calibrated: {line}")
        print(f"  Reference x-position: {self.seatpost_x}")
        return True

    def analyze_frame(self, frame, left_detected: Tuple[int, int], right_detected: Tuple[int, int]) -> Optional[Dict]:
        """
        Analyze one frame: track the two shoulder points, compute sway.
        
        Args:
            frame: video frame
            left_detected: detected left shoulder position (x, y)
            right_detected: detected right shoulder position (x, y)
        """
        h, w = frame.shape[:2]
        
        if not left_detected or not right_detected:
            return None
        
        left_x, left_y = left_detected
        right_x, right_y = right_detected
        
        # Midpoint between shoulders
        midpoint_x = (left_x + right_x) // 2
        midpoint_y = (left_y + right_y) // 2
        
        # Add to path history
        self.left_path.append((left_x, left_y))
        self.right_path.append((right_x, right_y))
        self.midpoint_path.append((midpoint_x, midpoint_y))
        
        # Compute deviation from seatpost
        deviation = midpoint_x - self.seatpost_x
        self.marker_xs.append(midpoint_x)
        
        # Smoothed deviation
        sway_smoothed = int(np.mean(list(self.marker_xs))) - self.seatpost_x if self.marker_xs else 0
        
        # Categorize direction
        threshold = 15
        if sway_smoothed > threshold:
            sway_direction = 'right'
        elif sway_smoothed < -threshold:
            sway_direction = 'left'
        else:
            sway_direction = 'center'
        
        self.frame_deviations.append(deviation)
        self.sway_sides.append(sway_direction)
        
        return {
            'left_x': left_x,
            'left_y': left_y,
            'right_x': right_x,
            'right_y': right_y,
            'midpoint_x': midpoint_x,
            'midpoint_y': midpoint_y,
            'seatpost_x': self.seatpost_x,
            'deviation': deviation,
            'sway_smoothed': sway_smoothed,
            'sway_direction': sway_direction,
        }

    def draw_overlay(self, frame: np.ndarray, data: Dict) -> np.ndarray:
        """Draw seatpost line, shoulder markers, paths, and sway visualization."""
        out = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw seatpost line
        if self.seatpost_line:
            cv2.line(out, self.seatpost_line[0], self.seatpost_line[1], (255, 255, 0), 3)
        
        left_x, left_y = data['left_x'], data['left_y']
        right_x, right_y = data['right_x'], data['right_y']
        midpoint_x, midpoint_y = data['midpoint_x'], data['midpoint_y']
        sway_smoothed = data['sway_smoothed']
        sway_direction = data['sway_direction']
        
        # Draw paths (fading)
        for i, (px, py) in enumerate(self.left_path):
            alpha = int(255 * (i / max(len(self.left_path), 1)))
            color = (255, 100, 100)  # Blue-ish
            cv2.circle(out, (px, py), 4, color, -1)
        
        for i, (px, py) in enumerate(self.right_path):
            alpha = int(255 * (i / max(len(self.right_path), 1)))
            color = (100, 100, 255)  # Red-ish
            cv2.circle(out, (px, py), 4, color, -1)
        
        # Draw current shoulder markers
        cv2.circle(out, (left_x, left_y), 10, (255, 0, 0), -1)
        cv2.circle(out, (left_x, left_y), 10, (255, 255, 255), 2)
        cv2.putText(out, "L", (left_x - 5, left_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.circle(out, (right_x, right_y), 10, (0, 0, 255), -1)
        cv2.circle(out, (right_x, right_y), 10, (255, 255, 255), 2)
        cv2.putText(out, "R", (right_x + 3, right_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw midpoint
        cv2.circle(out, (midpoint_x, midpoint_y), 8, (0, 255, 255), -1)
        cv2.circle(out, (midpoint_x, midpoint_y), 8, (255, 255, 255), 2)
        
        # Sway color & direction
        color = (0, 255, 255)
        if sway_direction == 'left':
            color = (255, 0, 0)
        elif sway_direction == 'right':
            color = (0, 0, 255)
        
        # Sway bar
        bar_y = midpoint_y
        bar_left = self.seatpost_x - 100
        bar_right = self.seatpost_x + 100
        cv2.line(out, (bar_left, bar_y), (bar_right, bar_y), (100, 100, 100), 1)
        cv2.line(out, (midpoint_x, bar_y - 5), (midpoint_x, bar_y + 5), color, 3)
        
        # Info
        cv2.putText(out, f"Sway: {sway_direction.upper()}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out, f"Deviation: {sway_smoothed:+.0f} px", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return out

    def get_statistics(self) -> Dict:
        """Compute sway statistics."""
        if not self.sway_sides:
            return {
                'total_frames': 0,
                'left_count': 0,
                'right_count': 0,
                'center_count': 0,
                'left_percent': 0.0,
                'right_percent': 0.0,
                'center_percent': 0.0,
                'avg_deviation': 0.0,
                'max_left_deviation': 0.0,
                'max_right_deviation': 0.0,
            }
        
        total = len(self.sway_sides)
        left_count = self.sway_sides.count('left')
        right_count = self.sway_sides.count('right')
        center_count = self.sway_sides.count('center')
        
        deviations = np.array(self.frame_deviations)
        
        return {
            'total_frames': total,
            'left_count': left_count,
            'right_count': right_count,
            'center_count': center_count,
            'left_percent': 100.0 * left_count / total if total > 0 else 0.0,
            'right_percent': 100.0 * right_count / total if total > 0 else 0.0,
            'center_percent': 100.0 * center_count / total if total > 0 else 0.0,
            'avg_deviation': float(np.mean(deviations)),
            'max_left_deviation': float(np.min(deviations)),
            'max_right_deviation': float(np.max(deviations)),
        }

    def reset(self):
        """Reset tracking for a new video/segment."""
        self.left_path.clear()
        self.right_path.clear()
        self.midpoint_path.clear()
        self.marker_xs.clear()
        self.frame_deviations = []
        self.frame_numbers = []
        self.sway_sides = []
