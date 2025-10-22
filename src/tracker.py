"""
Modern Object Tracking Module

This module provides a comprehensive object tracking solution with support for
multiple tracking algorithms, YOLO-based detection, and advanced features.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackerType(Enum):
    """Available tracker types."""
    CSRT = "CSRT"
    KCF = "KCF"
    MOSSE = "MOSSE"
    MIL = "MIL"
    BOOSTING = "BOOSTING"
    TLD = "TLD"
    MEDIANFLOW = "MEDIANFLOW"
    GOTURN = "GOTURN"


@dataclass
class TrackingResult:
    """Result of tracking operation."""
    success: bool
    bbox: Tuple[int, int, int, int]
    confidence: float = 0.0
    tracker_type: str = ""


class ObjectTracker:
    """
    Modern object tracker with support for multiple algorithms and features.
    
    This class provides a unified interface for various OpenCV tracking algorithms
    with enhanced error handling, logging, and state management.
    """
    
    def __init__(self, tracker_type: TrackerType = TrackerType.CSRT):
        """
        Initialize the object tracker.
        
        Args:
            tracker_type: Type of tracker to use
        """
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker()
        self.is_initialized = False
        self.tracking_history: List[TrackingResult] = []
        
    def _create_tracker(self) -> cv2.Tracker:
        """
        Create tracker instance based on type.
        
        Returns:
            OpenCV tracker instance
            
        Raises:
            ValueError: If tracker type is not supported
        """
        tracker_map = {
            TrackerType.CSRT: cv2.TrackerCSRT_create,
            TrackerType.KCF: cv2.TrackerKCF_create,
            TrackerType.MOSSE: cv2.TrackerMOSSE_create,
            TrackerType.MIL: cv2.TrackerMIL_create,
            TrackerType.BOOSTING: cv2.TrackerBoosting_create,
            TrackerType.TLD: cv2.TrackerTLD_create,
            TrackerType.MEDIANFLOW: cv2.TrackerMedianFlow_create,
            TrackerType.GOTURN: cv2.TrackerGOTURN_create,
        }
        
        if self.tracker_type not in tracker_map:
            raise ValueError(f"Unsupported tracker type: {self.tracker_type}")
            
        try:
            return tracker_map[self.tracker_type]()
        except Exception as e:
            logger.error(f"Failed to create {self.tracker_type.value} tracker: {e}")
            # Fallback to CSRT
            logger.info("Falling back to CSRT tracker")
            return cv2.TrackerCSRT_create()
    
    def initialize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize tracker with first frame and bounding box.
        
        Args:
            frame: First frame of the video
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            success = self.tracker.init(frame, bbox)
            self.is_initialized = success
            if success:
                logger.info(f"Tracker {self.tracker_type.value} initialized successfully")
            else:
                logger.error(f"Failed to initialize tracker {self.tracker_type.value}")
            return success
        except Exception as e:
            logger.error(f"Error initializing tracker: {e}")
            return False
    
    def update(self, frame: np.ndarray) -> TrackingResult:
        """
        Update tracker with new frame.
        
        Args:
            frame: Current frame
            
        Returns:
            TrackingResult with success status and bounding box
        """
        if not self.is_initialized:
            logger.warning("Tracker not initialized")
            return TrackingResult(False, (0, 0, 0, 0))
        
        try:
            success, bbox = self.tracker.update(frame)
            result = TrackingResult(
                success=success,
                bbox=tuple(map(int, bbox)) if success else (0, 0, 0, 0),
                tracker_type=self.tracker_type.value
            )
            self.tracking_history.append(result)
            return result
        except Exception as e:
            logger.error(f"Error updating tracker: {e}")
            return TrackingResult(False, (0, 0, 0, 0))
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.is_initialized = False
        self.tracking_history.clear()
        self.tracker = self._create_tracker()
        logger.info("Tracker reset")


class MultiObjectTracker:
    """
    Multi-object tracker for tracking multiple objects simultaneously.
    """
    
    def __init__(self, tracker_type: TrackerType = TrackerType.CSRT):
        """
        Initialize multi-object tracker.
        
        Args:
            tracker_type: Type of tracker to use for all objects
        """
        self.tracker_type = tracker_type
        self.trackers: Dict[int, ObjectTracker] = {}
        self.next_id = 0
        
    def add_object(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """
        Add new object to track.
        
        Args:
            frame: Current frame
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Object ID if successful, -1 otherwise
        """
        tracker = ObjectTracker(self.tracker_type)
        if tracker.initialize(frame, bbox):
            obj_id = self.next_id
            self.trackers[obj_id] = tracker
            self.next_id += 1
            logger.info(f"Added object {obj_id} to tracking")
            return obj_id
        return -1
    
    def update_all(self, frame: np.ndarray) -> Dict[int, TrackingResult]:
        """
        Update all trackers with new frame.
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary mapping object IDs to tracking results
        """
        results = {}
        for obj_id, tracker in self.trackers.items():
            results[obj_id] = tracker.update(frame)
        return results
    
    def remove_object(self, obj_id: int) -> bool:
        """
        Remove object from tracking.
        
        Args:
            obj_id: Object ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        if obj_id in self.trackers:
            del self.trackers[obj_id]
            logger.info(f"Removed object {obj_id} from tracking")
            return True
        return False
    
    def reset(self) -> None:
        """Reset all trackers."""
        self.trackers.clear()
        self.next_id = 0
        logger.info("Multi-object tracker reset")


def draw_tracking_result(
    frame: np.ndarray, 
    result: TrackingResult, 
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw tracking result on frame.
    
    Args:
        frame: Input frame
        result: Tracking result
        color: Bounding box color (BGR)
        thickness: Line thickness
        
    Returns:
        Frame with tracking visualization
    """
    if result.success:
        x, y, w, h = result.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            frame, 
            f"{result.tracker_type} Tracking", 
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            color, 
            2
        )
    else:
        cv2.putText(
            frame, 
            "Lost Tracking", 
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 0, 255), 
            2
        )
    return frame


def create_sample_video(output_path: str, duration: int = 10) -> None:
    """
    Create a sample video for testing purposes.
    
    Args:
        output_path: Path to save the video
        duration: Duration in seconds
    """
    # Video properties
    width, height = 640, 480
    fps = 30
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"Creating sample video: {output_path}")
    
    for frame_num in range(total_frames):
        # Create frame with moving circle
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving circle parameters
        center_x = int(width // 2 + 100 * np.sin(frame_num * 0.1))
        center_y = int(height // 2 + 50 * np.cos(frame_num * 0.1))
        radius = 30
        
        # Draw circle
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), -1)
        
        # Add some noise/background
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Sample video created: {output_path}")


if __name__ == "__main__":
    # Example usage
    tracker = ObjectTracker(TrackerType.CSRT)
    
    # Create sample video for testing
    sample_video_path = "data/sample_video.mp4"
    Path("data").mkdir(exist_ok=True)
    create_sample_video(sample_video_path)
    
    # Load video
    cap = cv2.VideoCapture(sample_video_path)
    
    # Read first frame
    success, frame = cap.read()
    if not success:
        print("Failed to read video")
        exit()
    
    # Select ROI
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False)
    cv2.destroyAllWindows()
    
    # Initialize tracker
    if tracker.initialize(frame, bbox):
        print("Tracker initialized successfully")
        
        # Track object
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            result = tracker.update(frame)
            frame = draw_tracking_result(frame, result)
            
            cv2.imshow("Object Tracking", frame)
            
            if cv2.waitKey(30) & 0xFF == 27:  # ESC key
                break
    
    cap.release()
    cv2.destroyAllWindows()
