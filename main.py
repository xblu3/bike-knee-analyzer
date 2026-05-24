#!/usr/bin/env python3
"""
Command-line interface for Bike Knee Analyzer.
"""

import argparse
import sys
from pathlib import Path

from bike_analyzer import BikeKneeAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video of a person on a stationary bike to measure knee bend angles."
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file to analyze'
    )
    
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Analyze live webcam feed'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save annotated output video'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to save CSV report with angle data'
    )
    
    parser.add_argument(
        '--plots',
        type=str,
        help='Directory to save analysis plots'
    )
    
    parser.add_argument(
        '--side',
        choices=['left', 'right'],
        default='right',
        help='Which leg to analyze (default: right)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display video during analysis'
    )
    
    parser.add_argument(
        '--complexity',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Model complexity: 0=lite, 1=full, 2=heavy (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video and not args.webcam:
        parser.error("Either --video or --webcam must be specified")
    
    if args.video and args.webcam:
        parser.error("Cannot use both --video and --webcam")
    
    # Initialize analyzer
    analyzer = BikeKneeAnalyzer(
        model_complexity=args.complexity,
        min_detection_confidence=0.5
    )
    
    try:
        if args.video:
            # Analyze video file
            video_path = Path(args.video)
            
            if not video_path.exists():
                print(f"Error: Video file not found: {args.video}", file=sys.stderr)
                sys.exit(1)
            
            print(f"Analyzing video: {args.video}")
            results = analyzer.analyze_video(
                str(video_path),
                side=args.side,
                output_video_path=args.output,
                show_visualization=args.show
            )
            
            # Print statistics
            print("\n" + "="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            stats = results['statistics']
            print(f"Frames analyzed: {stats['count']}")
            print(f"Minimum angle: {stats['min']:.2f}°")
            print(f"Maximum angle: {stats['max']:.2f}°")
            print(f"Average angle: {stats['avg']:.2f}°")
            print(f"Std deviation: {stats['std']:.2f}°")
            
            # Export CSV if requested
            if args.csv:
                analyzer.export_to_csv(args.csv)
                print(f"CSV report saved to: {args.csv}")
            
            # Generate plots if requested
            if args.plots:
                analyzer.generate_plots(args.plots)
                print(f"Plots saved to: {args.plots}")
            
            # Indicate output video
            if args.output:
                print(f"Annotated video saved to: {args.output}")
            
            print("="*50 + "\n")
        
        elif args.webcam:
            print("Webcam analysis not yet implemented")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
