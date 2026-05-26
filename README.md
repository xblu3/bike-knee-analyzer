# Bike Knee Analyzer

An application that analyzes video footage of a person on a stationary bike from a side view to measure and report knee bend angles in real-time.

## Features

- **Real-time Video Analysis**: Process video files or live webcam feed
- **Knee Bend Angle Detection**: Uses pose estimation to calculate knee angles
- **Frame-by-Frame Analysis**: Track angle changes throughout the pedal cycle
- **Statistics**: Min, max, and average knee angles
- **Visualization**: Overlay angle measurements on video output
- **Export Results**: Save analysis results as CSV or video with annotations

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Analyze a video file
python main.py --video path/to/video.mp4

# Analyze webcam feed
python main.py --webcam

# Save output video with annotations
python main.py --video path/to/video.mp4 --output output.mp4

# Generate CSV report
python main.py --video path/to/video.mp4 --csv report.csv
```

### Python API

```python
from bike_analyzer import BikeKneeAnalyzer

analyzer = BikeKneeAnalyzer()
results = analyzer.analyze_video('video.mp4')
print(f"Average knee angle: {results['avg_angle']}°")
print(f"Min angle: {results['min_angle']}°")
print(f"Max angle: {results['max_angle']}°")
```

## How It Works

1. **Pose Detection**: Uses MediaPipe Pose to detect 33 body landmarks
2. **Knee Identification**: Locates hip, knee, and ankle positions from landmarks
3. **Angle Calculation**: Calculates the angle at the knee joint using vector mathematics
4. **Tracking**: Monitors angle throughout the video frame by frame
5. **Analysis**: Computes statistics and generates visualizations

## Output

The analysis provides:
- Individual frame knee angles
- Statistics (min, max, average, std deviation)
- Frame-by-frame angle graph
- Annotated video showing angle at each frame
- CSV export for further analysis

## Project Structure

```
bike-knee-analyzer/
├── main.py                 # CLI entry point
├── bike_analyzer.py        # Core analysis engine
├── pose_detector.py        # MediaPipe wrapper
├── angle_calculator.py     # Geometry calculations
├── visualizer.py          # Video annotation and plotting
├── requirements.txt       # Python dependencies
└── tests/                 # Unit tests
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT
