"""
Object Tracking Package

A comprehensive object tracking solution with support for multiple algorithms,
YOLO-based detection, and modern Python features.
"""

__version__ = "1.0.0"
__author__ = "AI Projects Team"
__email__ = "ai-projects@example.com"

from .tracker import ObjectTracker, TrackerType, MultiObjectTracker, TrackingResult
from .yolo_tracker import YOLODetector, YOLOTracker, Detection
from .config import Config, config

__all__ = [
    "ObjectTracker",
    "TrackerType", 
    "MultiObjectTracker",
    "TrackingResult",
    "YOLODetector",
    "YOLOTracker",
    "Detection",
    "Config",
    "config"
]
