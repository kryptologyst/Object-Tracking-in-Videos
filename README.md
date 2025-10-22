# Object Tracking in Videos

A comprehensive object tracking solution with support for multiple algorithms, YOLO-based detection, and an intuitive web interface.

## Features

- **Multiple Tracking Algorithms**: CSRT, KCF, MOSSE, MIL, BOOSTING, TLD, MEDIANFLOW, GOTURN
- **YOLO Integration**: State-of-the-art object detection with ByteTrack for multi-object tracking
- **Web Interface**: User-friendly Streamlit app for easy interaction and demo
- **Modern Codebase**: Type hints, comprehensive docstrings, PEP8 compliance
- **Configuration Management**: YAML-based configuration system
- **Logging & Monitoring**: Comprehensive logging and tracking metrics
- **Sample Data**: Built-in sample video generation for testing

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Object-Tracking-in-Videos.git
cd Object-Tracking-in-Videos
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web interface:
```bash
streamlit run web_app/app.py
```

### Basic Usage

#### Command Line Interface

```python
from src.tracker import ObjectTracker, TrackerType

# Initialize tracker
tracker = ObjectTracker(TrackerType.CSRT)

# Load video and initialize
cap = cv2.VideoCapture('your_video.mp4')
success, frame = cap.read()
bbox = cv2.selectROI("Select Object", frame)
tracker.initialize(frame, bbox)

# Track object
while True:
    success, frame = cap.read()
    if not success:
        break
    
    result = tracker.update(frame)
    if result.success:
        # Draw bounding box
        x, y, w, h = result.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC
        break
```

#### YOLO-based Multi-Object Tracking

```python
from src.yolo_tracker import YOLOTracker

# Initialize YOLO tracker
tracker = YOLOTracker()

# Track objects in video
cap = cv2.VideoCapture('your_video.mp4')
while True:
    success, frame = cap.read()
    if not success:
        break
    
    detections = tracker.track(frame)
    frame = tracker.draw_tracks(frame, detections)
    
    cv2.imshow("YOLO Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break
```

## 📁 Project Structure

```
object-tracking-in-videos/
├── src/                    # Source code
│   ├── tracker.py         # Core tracking algorithms
│   ├── yolo_tracker.py    # YOLO-based tracking
│   └── config.py          # Configuration management
├── web_app/               # Streamlit web interface
│   └── app.py            # Main web application
├── config/               # Configuration files
│   └── config.yaml       # Default configuration
├── data/                 # Sample videos and datasets
├── models/               # Pre-trained models
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🔧 Configuration

The project uses YAML-based configuration. Edit `config/config.yaml` to customize:

```yaml
tracking:
  default_tracker: CSRT
  confidence_threshold: 0.5
  max_tracking_history: 100

yolo:
  model_path: yolov8n.pt
  device: cpu
  conf_threshold: 0.5
  iou_threshold: 0.45

video:
  default_fps: 30
  output_format: mp4v
  quality: high
```

## Web Interface

The Streamlit web interface provides:

- **Interactive Tracker Selection**: Choose from 8 different tracking algorithms
- **Video Upload**: Upload your own videos or use built-in samples
- **Real-time Visualization**: See tracking results in real-time
- **YOLO Integration**: Enable YOLO-based multi-object tracking
- **Configuration Panel**: Adjust parameters on the fly
- **Tracking Metrics**: Monitor tracking performance

### Running the Web App

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501`

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Tracking Algorithms

| Algorithm | Speed | Accuracy | Best For |
|-----------|-------|----------|----------|
| CSRT | Medium | High | General purpose, occlusion |
| KCF | Fast | Medium | Real-time applications |
| MOSSE | Very Fast | Medium | High-speed tracking |
| MIL | Slow | High | Appearance changes |
| BOOSTING | Slow | Medium | Classic applications |
| TLD | Medium | High | Learning-based tracking |
| MEDIANFLOW | Fast | Medium | Predictable motion |
| GOTURN | Medium | High | Deep learning-based |

## Advanced Features

### Multi-Object Tracking

```python
from src.tracker import MultiObjectTracker

# Initialize multi-object tracker
multi_tracker = MultiObjectTracker(TrackerType.CSRT)

# Add objects to track
obj_id1 = multi_tracker.add_object(frame, bbox1)
obj_id2 = multi_tracker.add_object(frame, bbox2)

# Update all trackers
results = multi_tracker.update_all(frame)
```

### Custom Video Generation

```python
from src.tracker import create_sample_video
from src.yolo_tracker import create_test_video_with_objects

# Create simple moving object video
create_sample_video("data/sample.mp4", duration=10)

# Create multi-object test video
create_test_video_with_objects("data/test.mp4", duration=15)
```

## 🛠️ Development

### Code Style

The project follows PEP8 standards. Format code with:

```bash
black src/ web_app/ tests/
```

Check code quality:

```bash
flake8 src/ web_app/ tests/
```

Type checking:

```bash
mypy src/ web_app/
```

### Adding New Trackers

1. Add tracker type to `TrackerType` enum in `src/tracker.py`
2. Add tracker creation logic in `_create_tracker()` method
3. Update web interface options in `web_app/app.py`

## Performance Tips

- **GPU Acceleration**: Use CUDA-enabled PyTorch for YOLO models
- **Video Optimization**: Use appropriate video codecs (H.264, H.265)
- **Tracker Selection**: Choose tracker based on speed vs accuracy requirements
- **Resolution**: Lower resolution videos process faster

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV team for excellent computer vision tools
- Ultralytics for YOLO implementation
- Supervision for tracking utilities
- Streamlit for web interface framework

## Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the documentation
- Review the example code


# Object-Tracking-in-Videos
