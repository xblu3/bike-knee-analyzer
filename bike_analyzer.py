"""
Core bike knee analyzer engine.
Supports front-view (side profile) knee angle analysis and rear-view body sway analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

from pose_detector import PoseDetector
from angle_calculator import calculate_knee_angle
from visualizer import VideoVisualizer, PlotVisualizer
from marker_detector import ColorMarkerDetector
from rear_view_analyzer import RearViewAnalyzer
from side_view_calibrator import PointCalibrator
from multi_angle_analyzer import MultiAngleAnalyzer

WINDOW_NAME = 'Bike Knee Analyzer'


class BikeKneeAnalyzer:
    """Main analyzer for bike riding: side-view knee angles & rear-view sway."""

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5):
        self.pose_detector = PoseDetector(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )
        self.angles = []
        self.frame_numbers = []

    def analyze_video(self,
                      video_path: str,
                      side: str = 'right',
                      output_video_path: Optional[str] = None,
                      show_visualization: bool = False,
                      window_width: int = 0,
                      start_sec: float = 0.0,
                      finish_sec: Optional[float] = None,
                      marker_detector: Optional[ColorMarkerDetector] = None,
                      rear_view: bool = False,
                      multi_angle: bool = False) -> Dict:
        """
        Analyze a video file for knee bend angles (side view) or body sway (rear view).

        Args:
            video_path:           Path to input video file
            side:                 'left' or 'right' leg (side-view mode only)
            output_video_path:    Path to save annotated video (optional)
            show_visualization:   Whether to display preview window
            window_width:         Initial preview window width (0 = native)
            start_sec:            Start time in seconds (default: 0)
            finish_sec:           Finish time in seconds (default: end)
            marker_detector:      ColorMarkerDetector instance (side-view only)
            rear_view:            If True, analyze body sway from rear
            multi_angle:          If True, analyze multiple joint angles (shoulder, elbow, hip, knee, ankle)

        Returns:
            Dictionary with analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        # Clamp time range to frame indices
        start_sec  = max(0.0, start_sec)
        if finish_sec is None or finish_sec > duration_sec:
            finish_sec = duration_sec
        start_frame   = int(start_sec  * fps)
        finish_frame  = int(finish_sec * fps)
        segment_frames = finish_frame - start_frame

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Output video writer
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height)
            )

        # Preview window
        if show_visualization:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            init_w = window_width if window_width > 0 else 1280
            init_h = 720
            cv2.resizeWindow(WINDOW_NAME, init_w, init_h)

        # ── Setup mode-specific analyzer ────────────────────────────────
        rear_analyzer = None
        multi_angle_analyzer = None
        calibrated_marker_points = None  # For side-view markers
        
        if rear_view:
            rear_analyzer = RearViewAnalyzer()
            
            # Calibration: get first frame and run calibrations
            ret_cal, frame_cal = cap.read()
            if ret_cal:
                # 1. User selects shoulder tracking points
                if not rear_analyzer.calibrate_shoulder_points(frame_cal):
                    raise ValueError("Shoulder point calibration cancelled")
                
                # 2. User draws seatpost line
                if not rear_analyzer.calibrate_seatpost_line(frame_cal):
                    raise ValueError("Seatpost calibration cancelled")
                
                # Seek back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            self.angles = []
            self.frame_numbers = []
        elif multi_angle:
            # Multi-angle mode: analyze all joints
            multi_angle_analyzer = MultiAngleAnalyzer(side=side)
            self.angles = []
            self.frame_numbers = []
            self.detection_sources = []
        else:
            self.angles = []
            self.frame_numbers = []
            self.detection_sources = []
            
            # Side-view: if using markers, calibrate 3 points (hip, knee, ankle)
            if marker_detector:
                ret_cal, frame_cal = cap.read()
                if ret_cal:
                    calibrator = PointCalibrator(
                        "Side-View Marker Calibration",
                        num_points=3,
                        point_names=["Hip", "Knee", "Ankle"]
                    )
                    calibrated_marker_points = calibrator.calibrate(frame_cal)
                    if calibrated_marker_points is None:
                        raise ValueError("Side-view marker calibration cancelled")
                    # Seek back to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        marker_hits = 0
        pose_hits = 0

        print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, "
              f"Duration: {duration_sec:.1f}s ({total_frames} frames)")
        print(f"Processing frames {start_frame}–{finish_frame} "
              f"({start_sec:.1f}s – {finish_sec:.1f}s, {segment_frames} frames)")
        mode_str = "rear-view body sway" if rear_view else "side-view knee angle"
        print(f"Analysis mode: {mode_str}")
        if show_visualization:
            print("Preview window open — press Q to quit early, drag corner to resize.")

        delay = max(1, int(1000 / fps) if fps > 0 else 30)

        # ── Main analysis loop ──────────────────────────────────────────
        while frame_count < segment_frames:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = frame.copy()

            if rear_view:
                # ── Rear-view: body sway analysis ──────────────────
                # Find the closest detected markers to calibrated shoulder points
                # Use color markers if available, otherwise use pose
                left_detected = None
                right_detected = None
                
                # Try color markers first
                if marker_detector:
                    positions = marker_detector.get_marker_positions(frame)
                    if positions:
                        markers = list(positions.values())
                        if len(markers) >= 2:
                            markers.sort(key=lambda p: p[0])
                            left_detected = markers[0]
                            right_detected = markers[1]
                
                # Fall back to pose detection
                if left_detected is None or right_detected is None:
                    landmarks = self.pose_detector.detect(frame)
                    if landmarks:
                        lm_list = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
                        # Use shoulders from pose (landmarks 11=left, 12=right)
                        sl = lm_list[11]
                        sr = lm_list[12]
                        sl_x = sl['x'] if isinstance(sl, dict) else sl.x
                        sl_y = sl['y'] if isinstance(sl, dict) else sl.y
                        sr_x = sr['x'] if isinstance(sr, dict) else sr.x
                        sr_y = sr['y'] if isinstance(sr, dict) else sr.y
                        
                        left_detected = (int(sl_x * w), int(sl_y * h))
                        right_detected = (int(sr_x * w), int(sr_y * h))
                
                # Analyze sway
                if left_detected and right_detected:
                    data = rear_analyzer.analyze_frame(frame, left_detected, right_detected)
                    if data:
                        annotated_frame = rear_analyzer.draw_overlay(frame, data)
            
            elif multi_angle:
                # ── Multi-angle analysis: shoulder, elbow, hip, knee, ankle ──
                landmarks = self.pose_detector.detect(frame)
                if landmarks:
                    data = multi_angle_analyzer.analyze_frame(landmarks, h, w)
                    if data:
                        annotated_frame = multi_angle_analyzer.draw_overlay(frame, data)
            else:
                # ── Side-view: knee angle analysis ──────────────────
                current_angle = None
                source = None

                # Try color markers first
                if marker_detector:
                    positions = marker_detector.get_marker_positions(frame)
                    if positions:
                        hip   = positions['hip']
                        knee  = positions['knee']
                        ankle = positions['ankle']
                        current_angle = calculate_knee_angle(hip, knee, ankle)
                        source = 'marker'
                        marker_hits += 1

                        annotated_frame = marker_detector.draw_markers(
                            annotated_frame, positions
                        )
                        annotated_frame = VideoVisualizer.draw_angle_on_frame(
                            annotated_frame, hip, knee, ankle, current_angle
                        )

                # Fall back to pose estimation
                if current_angle is None:
                    landmarks = self.pose_detector.detect(frame)
                    if landmarks:
                        knee_data = self.pose_detector.get_knee_landmarks(
                            landmarks, width, height, side=side
                        )
                        if knee_data:
                            hip   = knee_data['hip']
                            knee  = knee_data['knee']
                            ankle = knee_data['ankle']
                            current_angle = calculate_knee_angle(hip, knee, ankle)
                            source = 'pose'
                            pose_hits += 1

                            annotated_frame = VideoVisualizer.draw_angle_on_frame(
                                annotated_frame, hip, knee, ankle, current_angle
                            )

                # Record results
                if current_angle is not None:
                    self.angles.append(current_angle)
                    self.frame_numbers.append(start_frame + frame_count)
                    self.detection_sources.append(source)

                    stats = self.get_statistics()
                    stats['current'] = current_angle
                    stats['source'] = source
                    annotated_frame = VideoVisualizer.draw_statistics_on_frame(
                        annotated_frame, stats
                    )

            if video_writer:
                video_writer.write(annotated_frame)

            if show_visualization:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    print("Preview closed by user.")
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{segment_frames} frames")

        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if show_visualization:
            cv2.destroyWindow(WINDOW_NAME)
        self.pose_detector.close()

        if not rear_view and marker_detector:
            print(f"Detection summary: {marker_hits} marker frames, "
                  f"{pose_hits} pose-fallback frames")

        print(f"Analysis complete.")
        
        if rear_view:
            return self.get_rear_results(rear_analyzer)
        else:
            return self.get_results()

    # ------------------------------------------------------------------
    # Side-view (knee angle) results
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, float]:
        if not self.angles:
            return {'min': 0, 'max': 0, 'avg': 0, 'std': 0, 'count': 0}
        angles = np.array(self.angles)
        return {
            'min':   float(np.min(angles)),
            'max':   float(np.max(angles)),
            'avg':   float(np.mean(angles)),
            'std':   float(np.std(angles)),
            'count': len(angles),
        }

    def get_results(self) -> Dict:
        stats = self.get_statistics()
        return {
            'angles':                self.angles,
            'frame_numbers':         self.frame_numbers,
            'detection_sources':     getattr(self, 'detection_sources', []),
            'statistics':            stats,
            'min_angle':             stats['min'],
            'max_angle':             stats['max'],
            'avg_angle':             stats['avg'],
            'std_angle':             stats['std'],
            'total_frames_analyzed': stats['count'],
        }

    def export_to_csv(self, output_path: str):
        sources = getattr(self, 'detection_sources', [''] * len(self.angles))
        df = pd.DataFrame({
            'frame_number':       self.frame_numbers,
            'knee_angle_degrees': self.angles,
            'detection_source':   sources,
        })
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")

    # ------------------------------------------------------------------
    # Rear-view (body sway) results
    # ------------------------------------------------------------------

    def get_rear_results(self, rear_analyzer: RearViewAnalyzer) -> Dict:
        """Get rear-view analysis results."""
        stats = rear_analyzer.get_statistics()
        return {
            'statistics': stats,
            'mode': 'rear_view',
            'total_frames_analyzed': stats['total_frames'],
            'left_percent': stats['left_percent'],
            'right_percent': stats['right_percent'],
            'center_percent': stats['center_percent'],
            'avg_deviation': stats['avg_deviation'],
            'max_left_deviation': stats['max_left_deviation'],
            'max_right_deviation': stats['max_right_deviation'],
        }

    def export_rear_to_csv(self, output_path: str, rear_analyzer: RearViewAnalyzer):
        """Export rear-view analysis to CSV."""
        df = pd.DataFrame({
            'frame_number': rear_analyzer.frame_numbers,
            'sway_direction': rear_analyzer.sway_sides,
            'shoulder_deviation_px': rear_analyzer.frame_deviations,
        })
        df.to_csv(output_path, index=False)
        print(f"Rear-view results exported to: {output_path}")

    def generate_plots(self, output_dir: Optional[str] = None):
        if not self.angles:
            print("No data to plot")
            return

        plot_paths = {}
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            time_plot = str(Path(output_dir) / "knee_angle_timeline.png")
            hist_plot = str(Path(output_dir) / "knee_angle_histogram.png")
        else:
            time_plot = hist_plot = None

        print("Generating timeline plot...")
        PlotVisualizer.plot_angle_over_time(
            self.angles, self.frame_numbers,
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
