"""
YOLO-based Object Detection and Tracking Module

This module integrates YOLO object detection with tracking for more robust
multi-object tracking capabilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import torch
from ultralytics import YOLO
import supervision as sv

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Object detection result."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """
    YOLO-based object detector with support for various model sizes.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model or model name
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.device = device
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detections
        """
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detections.append(Detection(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def detect_specific_classes(
        self, 
        frame: np.ndarray, 
        target_classes: List[str], 
        conf_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Detect specific object classes.
        
        Args:
            frame: Input frame
            target_classes: List of class names to detect
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections for target classes only
        """
        all_detections = self.detect(frame, conf_threshold)
        return [det for det in all_detections if det.class_name in target_classes]


class YOLOTracker:
    """
    YOLO-based multi-object tracker using ByteTrack algorithm.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        Initialize YOLO tracker.
        
        Args:
            model_path: Path to YOLO model
            device: Device to run inference on
        """
        self.detector = YOLODetector(model_path, device)
        self.tracker = sv.ByteTrack()
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        
    def track(self, frame: np.ndarray, conf_threshold: float = 0.5) -> sv.Detections:
        """
        Track objects in frame.
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            
        Returns:
            Tracking results
        """
        try:
            # Detect objects
            detections = self.detector.detect(frame, conf_threshold)
            
            if not detections:
                return sv.Detections.empty()
            
            # Convert to supervision format
            boxes = np.array([det.bbox for det in detections])
            confidences = np.array([det.confidence for det in detections])
            class_ids = np.array([det.class_id for det in detections])
            
            sv_detections = sv.Detections(
                xyxy=sv.BoxAnnotator().xyxy_to_xywh(boxes),
                confidence=confidences,
                class_id=class_ids
            )
            
            # Update tracker
            tracked_detections = self.tracker.update_with_detections(sv_detections)
            
            # Update track history
            for track_id in tracked_detections.tracker_id:
                if track_id is not None:
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    # Get center point
                    x, y, w, h = tracked_detections.xyxy[tracked_detections.tracker_id == track_id][0]
                    center = (int(x + w/2), int(y + h/2))
                    self.track_history[track_id].append(center)
                    
                    # Keep only last 30 points
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id] = self.track_history[track_id][-30:]
            
            return tracked_detections
            
        except Exception as e:
            logger.error(f"Error in YOLO tracking: {e}")
            return sv.Detections.empty()
    
    def draw_tracks(
        self, 
        frame: np.ndarray, 
        detections: sv.Detections,
        show_trails: bool = True
    ) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            detections: Tracking detections
            show_trails: Whether to show tracking trails
            
        Returns:
            Frame with tracking visualization
        """
        annotated_frame = frame.copy()
        
        # Draw bounding boxes
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )
        
        # Draw labels
        labels = []
        for i, (confidence, class_id, tracker_id) in enumerate(
            zip(detections.confidence, detections.class_id, detections.tracker_id)
        ):
            if tracker_id is not None:
                labels.append(f"ID:{tracker_id} {self.detector.model.names[class_id]} {confidence:.2f}")
            else:
                labels.append(f"{self.detector.model.names[class_id]} {confidence:.2f}")
        
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )
        
        # Draw tracking trails
        if show_trails:
            trail_annotator = sv.TrailAnnotator()
            annotated_frame = trail_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
        
        return annotated_frame


def create_test_video_with_objects(output_path: str, duration: int = 15) -> None:
    """
    Create a test video with multiple moving objects.
    
    Args:
        output_path: Path to save the video
        duration: Duration in seconds
    """
    width, height = 800, 600
    fps = 30
    total_frames = duration * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.info(f"Creating test video with objects: {output_path}")
    
    for frame_num in range(total_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background
        frame[:] = (50, 50, 50)
        
        # Object 1: Moving circle (person-like)
        x1 = int(width * 0.2 + 200 * np.sin(frame_num * 0.05))
        y1 = int(height * 0.3 + 100 * np.cos(frame_num * 0.05))
        cv2.circle(frame, (x1, y1), 40, (0, 255, 0), -1)
        
        # Object 2: Moving rectangle (car-like)
        x2 = int(width * 0.6 + 150 * np.cos(frame_num * 0.03))
        y2 = int(height * 0.7 + 80 * np.sin(frame_num * 0.03))
        cv2.rectangle(frame, (x2-30, y2-20), (x2+30, y2+20), (255, 0, 0), -1)
        
        # Object 3: Moving triangle (bike-like)
        x3 = int(width * 0.4 + 100 * np.sin(frame_num * 0.07))
        y3 = int(height * 0.5 + 120 * np.cos(frame_num * 0.07))
        pts = np.array([[x3, y3-25], [x3-20, y3+15], [x3+20, y3+15]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 0, 255))
        
        # Add some noise
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    logger.info(f"Test video created: {output_path}")


if __name__ == "__main__":
    # Example usage
    try:
        # Create test video
        test_video_path = "data/test_objects.mp4"
        Path("data").mkdir(exist_ok=True)
        create_test_video_with_objects(test_video_path)
        
        # Initialize YOLO tracker
        tracker = YOLOTracker()
        
        # Load video
        cap = cv2.VideoCapture(test_video_path)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Track objects
            detections = tracker.track(frame)
            
            # Draw results
            frame = tracker.draw_tracks(frame, detections)
            
            cv2.imshow("YOLO Tracking", frame)
            
            if cv2.waitKey(30) & 0xFF == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Error in YOLO tracking demo: {e}")
        print("Note: YOLO tracking requires ultralytics and supervision packages")
        print("Install with: pip install ultralytics supervision")
