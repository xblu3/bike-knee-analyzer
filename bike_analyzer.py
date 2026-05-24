"""
Core bike knee analyzer engine.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

from pose_detector import PoseDetector
from angle_calculator import calculate_knee_angle
from visualizer import VideoVisualizer, PlotVisualizer


class BikeKneeAnalyzer:
    """Main analyzer for bike knee bend angles."""
    
    def __init__(self, model_complexity: int = 1, 
                 min_detection_confidence: float = 0.5):
        """
        Initialize the analyzer.
        
        Args:
            model_complexity: 0 (lite), 1 (full), 2 (heavy)
            min_detection_confidence: Minimum confidence threshold
        """
        self.pose_detector = PoseDetector(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )
        self.angles = []
        self.frame_numbers = []
    
    def analyze_video(self, video_path: str, 
                     side: str = 'right',
                     output_video_path: Optional[str] = None,
                     show_visualization: bool = False) -> Dict:
        """
        Analyze a video file for knee bend angles.
        
        Args:
            video_path: Path to input video file
            side: 'left' or 'right' leg
            output_video_path: Path to save annotated video (optional)
            show_visualization: Whether to display video during analysis
        
        Returns:
            Dictionary with analysis results
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path specified
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Reset angle tracking
        self.angles = []
        self.frame_numbers = []
        
        frame_count = 0
        
        print(f"Analyzing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect pose
            landmarks = self.pose_detector.detect(frame)
            
            annotated_frame = frame.copy()
            current_angle = None
            
            if landmarks:
                # Get knee landmarks
                knee_data = self.pose_detector.get_knee_landmarks(
                    landmarks, width, height, side=side
                )
                
                if knee_data:
                    # Calculate angle
                    hip = knee_data['hip']
                    knee = knee_data['knee']
                    ankle = knee_data['ankle']
                    
                    current_angle = calculate_knee_angle(hip, knee, ankle)
                    self.angles.append(current_angle)
                    self.frame_numbers.append(frame_count)
                    
                    # Annotate frame
                    annotated_frame = VideoVisualizer.draw_angle_on_frame(
                        annotated_frame, hip, knee, ankle, current_angle
                    )
            
            # Draw statistics
            if self.angles:
                stats = self.get_statistics()
                stats['current'] = current_angle or 0
                annotated_frame = VideoVisualizer.draw_statistics_on_frame(
                    annotated_frame, stats
                )
            
            # Write to output video
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display if requested
            if show_visualization:
                cv2.imshow('Bike Knee Analyzer', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        self.pose_detector.close()
        
        print(f"Analysis complete. Processed {len(self.angles)} frames with detected angles.")
        
        return self.get_results()
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics for detected angles.
        
        Returns:
            Dictionary with min, max, average, and std deviation
        """
        if not self.angles:
            return {
                'min': 0,
                'max': 0,
                'avg': 0,
                'std': 0,
                'count': 0
            }
        
        angles = np.array(self.angles)
        
        return {
            'min': float(np.min(angles)),
            'max': float(np.max(angles)),
            'avg': float(np.mean(angles)),
            'std': float(np.std(angles)),
            'count': len(angles)
        }
    
    def get_results(self) -> Dict:
        """
        Get complete analysis results.
        
        Returns:
            Dictionary with all analysis data
        """
        stats = self.get_statistics()
        
        return {
            'angles': self.angles,
            'frame_numbers': self.frame_numbers,
            'statistics': stats,
            'min_angle': stats['min'],
            'max_angle': stats['max'],
            'avg_angle': stats['avg'],
            'std_angle': stats['std'],
            'total_frames_analyzed': stats['count']
        }
    
    def export_to_csv(self, output_path: str):
        """
        Export analysis results to CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        df = pd.DataFrame({
            'frame_number': self.frame_numbers,
            'knee_angle_degrees': self.angles
        })
        
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
    
    def generate_plots(self, output_dir: Optional[str] = None):
        """
        Generate analysis plots.
        
        Args:
            output_dir: Directory to save plots (if None, display)
        """
        if not self.angles:
            print("No data to plot")
            return
        
        plot_paths = {}
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            time_plot = str(Path(output_dir) / "knee_angle_timeline.png")
            hist_plot = str(Path(output_dir) / "knee_angle_histogram.png")
        else:
            time_plot = None
            hist_plot = None
        
        print("Generating timeline plot...")
        PlotVisualizer.plot_angle_over_time(
            self.angles,
            self.frame_numbers,
            title="Knee Angle Throughout Video",
            save_path=time_plot
        )
        if time_plot:
            plot_paths['timeline'] = time_plot
        
        print("Generating histogram...")
        PlotVisualizer.plot_angle_histogram(
            self.angles,
            title="Knee Angle Distribution",
            save_path=hist_plot
        )
        if hist_plot:
            plot_paths['histogram'] = hist_plot
        
        return plot_paths
