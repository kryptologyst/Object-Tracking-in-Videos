#!/usr/bin/env python3
"""
Demo script for Object Tracking project.

This script demonstrates the various features of the object tracking system
including traditional trackers, YOLO-based tracking, and multi-object tracking.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import time

from src.tracker import ObjectTracker, TrackerType, MultiObjectTracker, create_sample_video
from src.yolo_tracker import YOLOTracker, create_test_video_with_objects

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_traditional_tracking():
    """Demonstrate traditional object tracking."""
    print("\n🎯 Demo 1: Traditional Object Tracking")
    print("=" * 50)
    
    # Create sample video
    video_path = "data/demo_sample.mp4"
    Path("data").mkdir(exist_ok=True)
    create_sample_video(video_path, duration=5)
    
    # Initialize tracker
    tracker = ObjectTracker(TrackerType.CSRT)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    
    if not success:
        print("❌ Failed to read video")
        return
    
    # Select ROI (simulate user selection)
    print("📹 Video loaded. Simulating object selection...")
    bbox = (200, 150, 80, 80)  # Simulated bounding box
    
    # Initialize tracker
    if tracker.initialize(frame, bbox):
        print("✅ Tracker initialized successfully")
        
        frame_count = 0
        success_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Update tracker
            result = tracker.update(frame)
            
            # Draw result
            if result.success:
                x, y, w, h = result.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Tracking: {result.tracker_type}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                success_count += 1
            else:
                cv2.putText(frame, "Lost Tracking", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Traditional Tracking Demo", frame)
            
            if cv2.waitKey(30) & 0xFF == 27:  # ESC
                break
            
            frame_count += 1
        
        success_rate = (success_count / frame_count) * 100 if frame_count > 0 else 0
        print(f"📊 Tracking Results: {success_count}/{frame_count} frames ({success_rate:.1f}% success)")
    
    cap.release()
    cv2.destroyAllWindows()


def demo_multi_object_tracking():
    """Demonstrate multi-object tracking."""
    print("\n🎯 Demo 2: Multi-Object Tracking")
    print("=" * 50)
    
    # Create test video with multiple objects
    video_path = "data/demo_multi.mp4"
    create_test_video_with_objects(video_path, duration=8)
    
    # Initialize multi-tracker
    multi_tracker = MultiObjectTracker(TrackerType.KCF)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    
    if not success:
        print("❌ Failed to read video")
        return
    
    print("📹 Multi-object video loaded. Adding objects to track...")
    
    # Add multiple objects (simulate user selections)
    bboxes = [
        (150, 200, 60, 60),  # Object 1
        (400, 300, 60, 60),  # Object 2
        (300, 100, 60, 60),  # Object 3
    ]
    
    object_ids = []
    for i, bbox in enumerate(bboxes):
        obj_id = multi_tracker.add_object(frame, bbox)
        if obj_id != -1:
            object_ids.append(obj_id)
            print(f"✅ Object {obj_id} added to tracking")
    
    if not object_ids:
        print("❌ No objects added to tracking")
        return
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Update all trackers
        results = multi_tracker.update_all(frame)
        
        # Draw results
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        
        for i, (obj_id, result) in enumerate(results.items()):
            if result.success:
                x, y, w, h = result.bbox
                color = colors[i % len(colors)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"Object {obj_id}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display frame
        cv2.imshow("Multi-Object Tracking Demo", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break
        
        frame_count += 1
    
    print(f"📊 Multi-object tracking completed: {frame_count} frames processed")
    
    cap.release()
    cv2.destroyAllWindows()


def demo_yolo_tracking():
    """Demonstrate YOLO-based tracking."""
    print("\n🎯 Demo 3: YOLO-based Tracking")
    print("=" * 50)
    
    try:
        # Initialize YOLO tracker
        tracker = YOLOTracker()
        print("✅ YOLO tracker initialized")
    except Exception as e:
        print(f"❌ Failed to initialize YOLO tracker: {e}")
        print("💡 Note: Install ultralytics and supervision packages for YOLO tracking")
        return
    
    # Create test video
    video_path = "data/demo_yolo.mp4"
    create_test_video_with_objects(video_path, duration=6)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Failed to open video")
        return
    
    print("📹 YOLO tracking video loaded")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Track objects
        detections = tracker.track(frame)
        
        # Draw results
        frame = tracker.draw_tracks(frame, detections)
        
        if len(detections) > 0:
            detection_count += 1
        
        # Display frame
        cv2.imshow("YOLO Tracking Demo", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:  # ESC
            break
        
        frame_count += 1
    
    detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
    print(f"📊 YOLO tracking completed: {detection_count}/{frame_count} frames with detections ({detection_rate:.1f}%)")
    
    cap.release()
    cv2.destroyAllWindows()


def demo_tracker_comparison():
    """Demonstrate different tracker algorithms."""
    print("\n🎯 Demo 4: Tracker Algorithm Comparison")
    print("=" * 50)
    
    # Create sample video
    video_path = "data/demo_comparison.mp4"
    create_sample_video(video_path, duration=3)
    
    # Test different trackers
    trackers_to_test = [
        TrackerType.CSRT,
        TrackerType.KCF,
        TrackerType.MOSSE,
    ]
    
    results = {}
    
    for tracker_type in trackers_to_test:
        print(f"\n🔄 Testing {tracker_type.value} tracker...")
        
        # Initialize tracker
        tracker = ObjectTracker(tracker_type)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        
        if not success:
            continue
        
        # Initialize with same bbox
        bbox = (200, 150, 80, 80)
        
        if not tracker.initialize(frame, bbox):
            print(f"❌ Failed to initialize {tracker_type.value}")
            cap.release()
            continue
        
        # Track and count successes
        success_count = 0
        frame_count = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            result = tracker.update(frame)
            if result.success:
                success_count += 1
            
            frame_count += 1
        
        success_rate = (success_count / frame_count) * 100 if frame_count > 0 else 0
        results[tracker_type.value] = success_rate
        
        print(f"✅ {tracker_type.value}: {success_count}/{frame_count} frames ({success_rate:.1f}% success)")
        
        cap.release()
    
    # Display comparison results
    print("\n📊 Tracker Comparison Results:")
    print("-" * 40)
    for tracker_name, success_rate in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{tracker_name:12}: {success_rate:6.1f}% success rate")


def main():
    """Run all demos."""
    print("🎯 Object Tracking Project Demo")
    print("=" * 60)
    print("This demo showcases various object tracking capabilities:")
    print("1. Traditional single-object tracking")
    print("2. Multi-object tracking")
    print("3. YOLO-based tracking")
    print("4. Tracker algorithm comparison")
    print("\nPress ESC in any window to skip to the next demo")
    print("=" * 60)
    
    # Run demos
    try:
        demo_traditional_tracking()
        time.sleep(1)
        
        demo_multi_object_tracking()
        time.sleep(1)
        
        demo_yolo_tracking()
        time.sleep(1)
        
        demo_tracker_comparison()
        
        print("\n🎉 All demos completed!")
        print("\n💡 Next steps:")
        print("- Run 'streamlit run web_app/app.py' for the web interface")
        print("- Run 'python cli.py --help' for command-line usage")
        print("- Check the README.md for detailed documentation")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
