#!/usr/bin/env python3
"""
Command-line interface for Bike Knee Analyzer.
Supports side-view knee angle analysis and rear-view body sway analysis.
"""

import argparse
import sys
from pathlib import Path

from bike_analyzer import BikeKneeAnalyzer
from marker_detector import ColorMarkerDetector, SUPPORTED_COLORS


def build_marker_detector(args) -> ColorMarkerDetector | None:
    """
    Return a ColorMarkerDetector if the user requested marker mode,
    or None for pure pose-estimation mode.
    """
    has_single = bool(args.marker_color)
    has_per    = bool(args.hip_color or args.knee_color or args.ankle_color)

    if not has_single and not has_per:
        return None     # marker mode not requested

    if has_per:
        missing = [j for j, v in [('hip',   args.hip_color),
                                   ('knee',  args.knee_color),
                                   ('ankle', args.ankle_color)] if not v]
        if missing:
            print(f"Error: when using per-marker colors you must supply all three: "
                  f"--hip-color, --knee-color, --ankle-color "
                  f"(missing: {', '.join(missing)})", file=sys.stderr)
            sys.exit(1)
        return ColorMarkerDetector(
            hip_color=args.hip_color,
            knee_color=args.knee_color,
            ankle_color=args.ankle_color,
        )

    # single color for all three
    return ColorMarkerDetector(marker_color=args.marker_color)


def build_rear_marker_detector(args) -> ColorMarkerDetector | None:
    """
    Return a ColorMarkerDetector configured for rear-view (2 markers: left & right shoulders).
    Returns None if markers not requested (will use pose fallback).
    """
    if args.left_side_marker and args.right_side_marker:
        # Use hip/knee slots to hold left/right shoulder markers
        return ColorMarkerDetector(
            hip_color=args.left_side_marker,
            knee_color=args.right_side_marker,
            ankle_color=args.left_side_marker,  # Dummy, won't be used
        )
    return None  # rear markers not requested, will use pose estimation


def main():
    parser = argparse.ArgumentParser(
        description="Analyze bike riding videos: side-view knee angles or rear-view body sway.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Input ──────────────────────────────────────────────────────────
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file to analyze')
    parser.add_argument('--start', type=float, default=0.0, metavar='SECONDS',
                        help='Start time in seconds (default: 0)')
    parser.add_argument('--finish', type=float, default=None, metavar='SECONDS',
                        help='Finish time in seconds (default: end of video)')

    # ── Analysis mode ──────────────────────────────────────────────────
    mode_group = parser.add_argument_group('analysis mode')
    mode_group.add_argument(
        '--rear-view', action='store_true',
        help='Analyze rear-view for body sway (default: side-view knee angles)'
    )
    mode_group.add_argument(
        '--multi-angle', action='store_true',
        help='Analyze multiple joint angles: shoulder, elbow, hip, knee, ankle (default: knee only)'
    )

    # ── Marker mode (side-view only) ───────────────────────────────────
    marker_group = parser.add_argument_group(
        'marker mode (side-view: knee angles)',
        'Use physical colored markers instead of (or in addition to) pose estimation.\n'
        f'Supported colors: {", ".join(SUPPORTED_COLORS)}'
    )
    marker_group.add_argument(
        '--marker-color', type=str, metavar='COLOR',
        help='Single color shared by all three markers (hip, knee, ankle)'
    )
    marker_group.add_argument(
        '--hip-color',   type=str, metavar='COLOR',
        help='Color of the HIP marker (requires --knee-color and --ankle-color too)'
    )
    marker_group.add_argument(
        '--knee-color',  type=str, metavar='COLOR',
        help='Color of the KNEE marker'
    )
    marker_group.add_argument(
        '--ankle-color', type=str, metavar='COLOR',
        help='Color of the ANKLE marker'
    )

    # ── Rear-view markers (rear-view only, OPTIONAL) ────────────────────
    rear_marker_group = parser.add_argument_group(
        'rear-view markers (rear-view: body sway, OPTIONAL)',
        'Optional: Use color markers to aid detection of shoulders.\n'
        'If not provided, pose estimation will be used as fallback.\n'
        f'Supported colors: {", ".join(SUPPORTED_COLORS)}'
    )
    rear_marker_group.add_argument(
        '--left-side-marker', type=str, metavar='COLOR',
        help='Color of LEFT shoulder marker (optional for --rear-view)'
    )
    rear_marker_group.add_argument(
        '--right-side-marker', type=str, metavar='COLOR',
        help='Color of RIGHT shoulder marker (optional for --rear-view)'
    )

    # ── Output ─────────────────────────────────────────────────────────
    parser.add_argument('--output', type=str,
                        help='Path to save annotated output video')
    parser.add_argument('--csv', type=str,
                        help='Path to save CSV report with results')
    parser.add_argument('--plots', type=str,
                        help='Directory to save analysis plots (side-view only)')

    # ── Display & pose ─────────────────────────────────────────────────
    parser.add_argument('--side', choices=['left', 'right'], default='right',
                        help='Which leg to analyze via pose (side-view only; default: right)')
    parser.add_argument('--show', action='store_true',
                        help='Display resizable preview window during analysis')
    parser.add_argument('--window-width', type=int, default=0,
                        help='Initial preview window width in pixels (0 = video width)')
    parser.add_argument('--complexity', type=int, choices=[0, 1, 2], default=1,
                        help='Pose model complexity: 0=lite, 1=full, 2=heavy (default: 1)')

    args = parser.parse_args()

    # ── Validate ───────────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    if args.start < 0:
        print("Error: --start must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.finish is not None and args.finish <= args.start:
        print("Error: --finish must be greater than --start", file=sys.stderr)
        sys.exit(1)

    # Marker colors only valid in side-view knee mode
    has_marker_args = (args.marker_color or args.hip_color or 
                       args.knee_color or args.ankle_color)
    if (args.rear_view or args.multi_angle) and has_marker_args:
        print("Error: --marker-color and per-marker colors are for side-view knee mode. "
              "Use --left-side-marker and --right-side-marker for rear-view", file=sys.stderr)
        sys.exit(1)

    # Rear-view no longer requires markers (optional)
    if args.rear_view:
        if (args.left_side_marker and not args.right_side_marker) or \
           (args.right_side_marker and not args.left_side_marker):
            print("Error: both --left-side-marker and --right-side-marker must be provided together",
                  file=sys.stderr)
            sys.exit(1)

    # Validate marker color names
    all_color_args = [('--marker-color', args.marker_color),
                      ('--hip-color',    args.hip_color),
                      ('--knee-color',   args.knee_color),
                      ('--ankle-color',  args.ankle_color),
                      ('--left-side-marker',  args.left_side_marker),
                      ('--right-side-marker', args.right_side_marker)]
    
    for flag, val in all_color_args:
        if val and val.lower() not in SUPPORTED_COLORS:
            print(f"Error: {flag} '{val}' is not supported. "
                  f"Choose from: {', '.join(SUPPORTED_COLORS)}", file=sys.stderr)
            sys.exit(1)

    marker_detector = None
    if args.rear_view:
        marker_detector = build_rear_marker_detector(args)
    else:
        marker_detector = build_marker_detector(args)

    # ── Run ────────────────────────────────────────────────────────────
    analyzer = BikeKneeAnalyzer(
        model_complexity=args.complexity,
        min_detection_confidence=0.5
    )

    try:
        print(f"Analyzing video: {args.video}")
        if args.start > 0 or args.finish is not None:
            finish_label = f"{args.finish}s" if args.finish is not None else "end"
            print(f"Time range: {args.start}s → {finish_label}")

        if args.rear_view:
            print("📷 Rear-view mode: analyzing body sway")
            if marker_detector:
                colors = marker_detector.per_marker_colors
                print(f"Left shoulder marker:  {colors['hip']}")
                print(f"Right shoulder marker: {colors['knee']}")
            else:
                print("Using pose estimation for shoulder detection (markers optional)")
        elif args.multi_angle:
            print("📷 Multi-angle mode: analyzing shoulder, elbow, hip, knee, ankle angles")
        else:
            print("📷 Side-view mode: analyzing knee angles")
            if marker_detector:
                if marker_detector.per_marker_colors:
                    colors = marker_detector.per_marker_colors
                    print(f"Marker colors — hip: {colors['hip']}, "
                          f"knee: {colors['knee']}, ankle: {colors['ankle']}")
                else:
                    print(f"Marker color: {marker_detector.single_color} (all three joints)")

        results = analyzer.analyze_video(
            str(video_path),
            side=args.side,
            output_video_path=args.output,
            show_visualization=args.show,
            window_width=args.window_width,
            start_sec=args.start,
            finish_sec=args.finish,
            marker_detector=marker_detector,
            rear_view=args.rear_view,
            multi_angle=args.multi_angle,
        )

        # ── Print results ──────────────────────────────────────────────
        print("\n" + "=" * 50)
        if args.rear_view:
            print("REAR-VIEW ANALYSIS RESULTS")
            print("=" * 50)
            stats = results['statistics']
            print(f"Frames analyzed:     {stats['total_frames']}")
            print(f"Left sway:           {stats['left_percent']:.1f}%")
            print(f"Right sway:          {stats['right_percent']:.1f}%")
            print(f"Center (neutral):    {stats['center_percent']:.1f}%")
            print(f"Avg deviation:       {stats['avg_deviation']:+.1f} px")
            print(f"Max left:            {stats['max_left_deviation']:+.1f} px")
            print(f"Max right:           {stats['max_right_deviation']:+.1f} px")
        else:
            print("SIDE-VIEW ANALYSIS RESULTS")
            print("=" * 50)
            stats = results['statistics']
            print(f"Frames analyzed:     {stats['count']}")
            print(f"Minimum angle:       {stats['min']:.2f}°")
            print(f"Maximum angle:       {stats['max']:.2f}°")
            print(f"Average angle:       {stats['avg']:.2f}°")
            print(f"Std deviation:       {stats['std']:.2f}°")

            sources = results.get('detection_sources', [])
            if sources:
                n_marker = sources.count('marker')
                n_pose   = sources.count('pose')
                if n_marker or n_pose:
                    print(f"Detected by:         markers={n_marker}, pose={n_pose}")

        if args.csv:
            if args.rear_view:
                # Need access to rear_analyzer for rear-view export
                # For now, just indicate where it would be saved
                print(f"CSV report would be saved to: {args.csv}")
            else:
                analyzer.export_to_csv(args.csv)
                print(f"CSV report saved to: {args.csv}")

        if args.plots and not args.rear_view:
            analyzer.generate_plots(args.plots)
            print(f"Plots saved to: {args.plots}")

        if args.output:
            print(f"Annotated video saved to: {args.output}")

        print("=" * 50 + "\n")

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
