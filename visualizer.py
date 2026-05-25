"""
Visualization utilities for annotating videos and generating plots.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


class VideoVisualizer:
    """Handle video annotation and visualization."""
    
    @staticmethod
    def create_resizable_window(window_name: str = 'Bike Knee Analyzer'):
        """Create a resizable window for video preview."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)  # Default size
    
    @staticmethod
    def draw_angle_on_frame(frame: np.ndarray, 
                           hip: tuple, 
                           knee: tuple, 
                           ankle: tuple,
                           angle: float,
                           color: tuple = (0, 255, 0),
                           thickness: int = 2) -> np.ndarray:
        """
        Draw angle visualization on a frame.
        
        Args:
            frame: BGR image frame
            hip: (x, y) hip position
            knee: (x, y) knee position
            ankle: (x, y) ankle position
            angle: Angle value in degrees
            color: BGR color tuple
            thickness: Line thickness
        
        Returns:
            Annotated frame
        """
        frame = frame.copy()
        
        # Convert to integer coordinates
        hip_pt = tuple(map(int, hip))
        knee_pt = tuple(map(int, knee))
        ankle_pt = tuple(map(int, ankle))
        
        # Draw lines for the angle
        cv2.line(frame, hip_pt, knee_pt, color, thickness)
        cv2.line(frame, knee_pt, ankle_pt, color, thickness)
        
        # Draw circles at joints
        cv2.circle(frame, hip_pt, 5, color, -1)
        cv2.circle(frame, knee_pt, 5, color, -1)
        cv2.circle(frame, ankle_pt, 5, color, -1)
        
        # Draw angle text
        text = f"{angle:.1f}°"
        text_pos = (knee_pt[0] + 20, knee_pt[1] - 20)
        cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, color, 2, cv2.LINE_AA)
        
        return frame
    
    @staticmethod
    def draw_statistics_on_frame(frame: np.ndarray,
                                stats: Dict[str, float]) -> np.ndarray:
        """
        Draw statistics text on frame.
        
        Args:
            frame: BGR image frame
            stats: Dictionary with 'current', 'min', 'max', 'avg' angles
        
        Returns:
            Annotated frame
        """
        frame = frame.copy()
        
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)
        
        texts = [
            f"Current: {stats.get('current', 0):.1f}°",
            f"Min: {stats.get('min', 0):.1f}°",
            f"Max: {stats.get('max', 0):.1f}°",
            f"Avg: {stats.get('avg', 0):.1f}°"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (10, y_offset + i * 25), 
                       font, font_scale, color, thickness)
        
        return frame


class PlotVisualizer:
    """Generate plots and graphs for analysis results."""
    
    @staticmethod
    def plot_angle_over_time(angles: List[float], 
                            frame_numbers: Optional[List[int]] = None,
                            fps: float = 30.0,
                            title: str = "Knee Angle Over Time",
                            save_path: Optional[str] = None):
        """
        Plot knee angle changes over time.
        
        Args:
            angles: List of angle values
            frame_numbers: List of frame numbers (if None, use 0-based index)
            fps: Frames per second for time calculation
            title: Plot title
            save_path: Path to save the figure (if None, display)
        """
        if frame_numbers is None:
            frame_numbers = list(range(len(angles)))
        
        # Convert frame numbers to time in seconds
        time_seconds = [f / fps for f in frame_numbers]
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_seconds, angles, 'b-', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Knee Angle (degrees)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add statistics to plot
        min_angle = min(angles)
        max_angle = max(angles)
        avg_angle = sum(angles) / len(angles)
        
        stats_text = f'Min: {min_angle:.1f}°\nMax: {max_angle:.1f}°\nAvg: {avg_angle:.1f}°'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_angle_histogram(angles: List[float],
                            title: str = "Knee Angle Distribution",
                            save_path: Optional[str] = None,
                            bins: int = 20):
        """
        Plot histogram of knee angles.
        
        Args:
            angles: List of angle values
            title: Plot title
            save_path: Path to save the figure
            bins: Number of histogram bins
        """
        plt.figure(figsize=(10, 6))
        plt.hist(angles, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Knee Angle (degrees)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
