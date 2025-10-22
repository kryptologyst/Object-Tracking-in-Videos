#!/usr/bin/env python3
"""
Command Line Interface for Object Tracking

This module provides a command-line interface for object tracking
with various options and configurations.
"""

import argparse
import cv2
import sys
from pathlib import Path
import logging
from typing import Optional

from src.tracker import ObjectTracker, TrackerType, MultiObjectTracker
from src.yolo_tracker import YOLOTracker
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Object Tracking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic tracking with CSRT
  python cli.py --video sample.mp4 --tracker CSRT
  
  # YOLO-based multi-object tracking
  python cli.py --video sample.mp4 --yolo
  
  # Multi-object tracking with custom confidence
  python cli.py --video sample.mp4 --multi-object --tracker KCF
  
  # Save output video
  python cli.py --video sample.mp4 --tracker CSRT --output result.mp4
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Path to input video file'
    )
    
    # Tracker selection
    tracker_group = parser.add_mutually_exclusive_group()
    tracker_group.add_argument(
        '--tracker', '-t',
        choices=[t.value for t in TrackerType],
        default='CSRT',
        help='Tracker algorithm to use'
    )
    tracker_group.add_argument(
        '--yolo',
        action='store_true',
        help='Use YOLO-based tracking'
    )
    
    # Multi-object tracking
    parser.add_argument(
        '--multi-object', '-m',
        action='store_true',
        help='Enable multi-object tracking'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Path to save output video'
    )
    
    # YOLO-specific options
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for YOLO detection'
    )
    
    parser.add_argument(
        '--model-path',
        default='yolov8n.pt',
        help='Path to YOLO model'
    )
    
    # Display options
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (useful for headless operation)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video FPS'
    )
    
    # Debug options
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def validate_video_path(video_path: str) -> bool:
    """Validate video file path."""
    path = Path(video_path)
    if not path.exists():
        logger.error(f"Video file not found: {video_path}")
        return False
    
    if not path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        logger.warning(f"Unusual video format: {path.suffix}")
    
    return True


def setup_video_writer(output_path: str, fps: int, frame_size: tuple) -> Optional[cv2.VideoWriter]:
    """Setup video writer for output."""
    if not output_path:
        return None
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    if not writer.isOpened():
        logger.error(f"Failed to create video writer: {output_path}")
        return None
    
    logger.info(f"Video writer created: {output_path}")
    return writer


def run_single_object_tracking(
    video_path: str,
    tracker_type: str,
    output_path: Optional[str] = None,
    display: bool = True,
    fps: int = 30
) -> None:
    """Run single object tracking."""
    logger.info(f"Starting single object tracking with {tracker_type}")
    
    # Initialize tracker
    tracker_enum = TrackerType(tracker_type)
    tracker = ObjectTracker(tracker_enum)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {width}x{height} @ {video_fps} FPS")
    
    # Setup video writer
    writer = setup_video_writer(output_path, fps, (width, height))
    
    # Read first frame and initialize tracker
    success, frame = cap.read()
    if not success:
        logger.error("Failed to read first frame")
        return
    
    # Select ROI
    logger.info("Select object to track and press Enter")
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False)
    cv2.destroyAllWindows()
    
    if bbox == (0, 0, 0, 0):
        logger.error("No object selected")
        return
    
    # Initialize tracker
    if not tracker.initialize(frame, bbox):
        logger.error("Failed to initialize tracker")
        return
    
    logger.info("Tracker initialized successfully")
    
    # Process video
    frame_count = 0
    tracking_success_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Update tracker
        result = tracker.update(frame)
        
        # Draw tracking result
        if result.success:
            x, y, w, h = result.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{result.tracker_type} Tracking",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            tracking_success_count += 1
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
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        # Display frame
        if display:
            cv2.imshow("Object Tracking", frame)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC key
                break
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # Print statistics
    success_rate = (tracking_success_count / frame_count) * 100 if frame_count > 0 else 0
    logger.info(f"Tracking completed: {tracking_success_count}/{frame_count} frames ({success_rate:.1f}% success)")


def run_yolo_tracking(
    video_path: str,
    output_path: Optional[str] = None,
    display: bool = True,
    fps: int = 30,
    conf_threshold: float = 0.5,
    model_path: str = "yolov8n.pt"
) -> None:
    """Run YOLO-based multi-object tracking."""
    logger.info("Starting YOLO-based multi-object tracking")
    
    try:
        # Initialize YOLO tracker
        tracker = YOLOTracker(model_path)
    except Exception as e:
        logger.error(f"Failed to initialize YOLO tracker: {e}")
        logger.error("Make sure ultralytics and supervision are installed")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {width}x{height}")
    
    # Setup video writer
    writer = setup_video_writer(output_path, fps, (width, height))
    
    # Process video
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Track objects
        detections = tracker.track(frame)
        
        # Draw tracking results
        frame = tracker.draw_tracks(frame, detections)
        
        # Write frame to output video
        if writer:
            writer.write(frame)
        
        # Display frame
        if display:
            cv2.imshow("YOLO Tracking", frame)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC key
                break
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    logger.info(f"YOLO tracking completed: {frame_count} frames processed")


def main():
    """Main CLI function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate video path
    if not validate_video_path(args.video):
        sys.exit(1)
    
    # Run appropriate tracking method
    try:
        if args.yolo:
            run_yolo_tracking(
                video_path=args.video,
                output_path=args.output,
                display=not args.no_display,
                fps=args.fps,
                conf_threshold=args.conf_threshold,
                model_path=args.model_path
            )
        else:
            run_single_object_tracking(
                video_path=args.video,
                tracker_type=args.tracker,
                output_path=args.output,
                display=not args.no_display,
                fps=args.fps
            )
    
    except KeyboardInterrupt:
        logger.info("Tracking interrupted by user")
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        sys.exit(1)
    
    logger.info("Tracking completed successfully")


if __name__ == "__main__":
    main()
