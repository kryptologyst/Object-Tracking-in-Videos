"""
Unit tests for object tracking project.
"""

import pytest
import cv2
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.tracker import ObjectTracker, TrackerType, MultiObjectTracker, TrackingResult
from src.yolo_tracker import YOLODetector, YOLOTracker, Detection
from src.config import Config


class TestObjectTracker:
    """Test cases for ObjectTracker class."""
    
    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = ObjectTracker(TrackerType.CSRT)
        assert tracker.tracker_type == TrackerType.CSRT
        assert not tracker.is_initialized
        assert len(tracker.tracking_history) == 0
    
    def test_tracker_creation(self):
        """Test tracker creation for different types."""
        for tracker_type in TrackerType:
            tracker = ObjectTracker(tracker_type)
            assert tracker.tracker is not None
    
    def test_tracker_initialization_with_frame(self):
        """Test tracker initialization with frame and bbox."""
        tracker = ObjectTracker(TrackerType.CSRT)
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 50, 50)
        
        success = tracker.initialize(frame, bbox)
        assert success
        assert tracker.is_initialized
    
    def test_tracker_update(self):
        """Test tracker update functionality."""
        tracker = ObjectTracker(TrackerType.CSRT)
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 50, 50)
        
        # Initialize tracker
        tracker.initialize(frame, bbox)
        
        # Update with new frame
        new_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.update(new_frame)
        
        assert isinstance(result, TrackingResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'bbox')
        assert hasattr(result, 'tracker_type')
    
    def test_tracker_reset(self):
        """Test tracker reset functionality."""
        tracker = ObjectTracker(TrackerType.CSRT)
        
        # Initialize tracker
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 50, 50)
        tracker.initialize(frame, bbox)
        
        # Reset tracker
        tracker.reset()
        
        assert not tracker.is_initialized
        assert len(tracker.tracking_history) == 0


class TestMultiObjectTracker:
    """Test cases for MultiObjectTracker class."""
    
    def test_multi_tracker_initialization(self):
        """Test multi-object tracker initialization."""
        multi_tracker = MultiObjectTracker(TrackerType.CSRT)
        assert len(multi_tracker.trackers) == 0
        assert multi_tracker.next_id == 0
    
    def test_add_object(self):
        """Test adding object to multi-tracker."""
        multi_tracker = MultiObjectTracker(TrackerType.CSRT)
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 50, 50)
        
        obj_id = multi_tracker.add_object(frame, bbox)
        assert obj_id == 0
        assert len(multi_tracker.trackers) == 1
    
    def test_update_all(self):
        """Test updating all trackers."""
        multi_tracker = MultiObjectTracker(TrackerType.CSRT)
        
        # Add objects
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox1 = (100, 100, 50, 50)
        bbox2 = (200, 200, 50, 50)
        
        multi_tracker.add_object(frame, bbox1)
        multi_tracker.add_object(frame, bbox2)
        
        # Update all
        new_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = multi_tracker.update_all(new_frame)
        
        assert len(results) == 2
        assert all(isinstance(result, TrackingResult) for result in results.values())
    
    def test_remove_object(self):
        """Test removing object from multi-tracker."""
        multi_tracker = MultiObjectTracker(TrackerType.CSRT)
        
        # Add object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 50, 50)
        obj_id = multi_tracker.add_object(frame, bbox)
        
        # Remove object
        success = multi_tracker.remove_object(obj_id)
        assert success
        assert len(multi_tracker.trackers) == 0


class TestYOLODetector:
    """Test cases for YOLODetector class."""
    
    @patch('src.yolo_tracker.YOLO')
    def test_yolo_detector_initialization(self, mock_yolo):
        """Test YOLO detector initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector("yolov8n.pt", "cpu")
        assert detector.device == "cpu"
        assert detector.model == mock_model
    
    @patch('src.yolo_tracker.YOLO')
    def test_yolo_detection(self, mock_yolo):
        """Test YOLO detection functionality."""
        # Mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        
        mock_box.xyxy = [np.array([100, 100, 150, 150])]
        mock_box.conf = [np.array([0.8])]
        mock_box.cls = [np.array([0])]
        
        mock_result.boxes = mock_box
        mock_model.return_value = [mock_result]
        mock_model.names = {0: 'person'}
        
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector("yolov8n.pt", "cpu")
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detections = detector.detect(frame)
        assert len(detections) == 1
        assert isinstance(detections[0], Detection)


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_initialization(self):
        """Test config initialization."""
        config = Config()
        assert config.config is not None
    
    def test_config_get(self):
        """Test config get method."""
        config = Config()
        value = config.get('tracking.default_tracker')
        assert value is not None
    
    def test_config_set(self):
        """Test config set method."""
        config = Config()
        config.set('test.key', 'test_value')
        value = config.get('test.key')
        assert value == 'test_value'
    
    def test_config_save(self):
        """Test config save method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            config = Config(config_path)
            config.set('test.key', 'test_value')
            config.save()
            
            # Verify file was created
            assert os.path.exists(config_path)
        finally:
            os.unlink(config_path)


class TestTrackingResult:
    """Test cases for TrackingResult dataclass."""
    
    def test_tracking_result_creation(self):
        """Test TrackingResult creation."""
        result = TrackingResult(
            success=True,
            bbox=(100, 100, 50, 50),
            confidence=0.8,
            tracker_type="CSRT"
        )
        
        assert result.success is True
        assert result.bbox == (100, 100, 50, 50)
        assert result.confidence == 0.8
        assert result.tracker_type == "CSRT"


class TestDetection:
    """Test cases for Detection dataclass."""
    
    def test_detection_creation(self):
        """Test Detection creation."""
        detection = Detection(
            bbox=(100, 100, 50, 50),
            confidence=0.8,
            class_id=0,
            class_name="person"
        )
        
        assert detection.bbox == (100, 100, 50, 50)
        assert detection.confidence == 0.8
        assert detection.class_id == 0
        assert detection.class_name == "person"


if __name__ == "__main__":
    pytest.main([__file__])
