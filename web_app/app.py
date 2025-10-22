"""
Streamlit Web Interface for Object Tracking

This module provides a user-friendly web interface for object tracking
with various algorithms and real-time visualization.
"""

import streamlit as st
import cv2
import numpy as np
from typing import Optional, List, Tuple
import tempfile
import os
from pathlib import Path
import logging

# Import our tracking modules
from tracker import ObjectTracker, TrackerType, MultiObjectTracker, create_sample_video
from yolo_tracker import YOLOTracker, create_test_video_with_objects

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Object Tracking Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'tracker' not in st.session_state:
        st.session_state.tracker = None
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'tracking_results' not in st.session_state:
        st.session_state.tracking_results = []
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = None

def create_sample_videos():
    """Create sample videos for demo."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_video_path = data_dir / "sample_video.mp4"
    test_video_path = data_dir / "test_objects.mp4"
    
    if not sample_video_path.exists():
        with st.spinner("Creating sample video..."):
            create_sample_video(str(sample_video_path))
    
    if not test_video_path.exists():
        with st.spinner("Creating test video with objects..."):
            create_test_video_with_objects(str(test_video_path))
    
    return sample_video_path, test_video_path

def process_video_frame(frame: np.ndarray, tracker_type: str, use_yolo: bool = False) -> np.ndarray:
    """
    Process a single video frame with tracking.
    
    Args:
        frame: Input frame
        tracker_type: Type of tracker to use
        use_yolo: Whether to use YOLO-based tracking
        
    Returns:
        Processed frame with tracking visualization
    """
    try:
        if use_yolo:
            # Use YOLO tracker
            if st.session_state.tracker is None:
                st.session_state.tracker = YOLOTracker()
            
            detections = st.session_state.tracker.track(frame)
            return st.session_state.tracker.draw_tracks(frame, detections)
        
        else:
            # Use traditional tracker
            if st.session_state.tracker is None:
                tracker_enum = TrackerType(tracker_type)
                st.session_state.tracker = ObjectTracker(tracker_enum)
            
            result = st.session_state.tracker.update(frame)
            
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
    
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return frame

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Object Tracking Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Tracker selection
    tracker_type = st.sidebar.selectbox(
        "Select Tracker Type",
        ["CSRT", "KCF", "MOSSE", "MIL", "BOOSTING", "TLD", "MEDIANFLOW", "GOTURN"],
        help="Choose the tracking algorithm to use"
    )
    
    # YOLO option
    use_yolo = st.sidebar.checkbox(
        "Use YOLO-based Tracking",
        help="Enable YOLO object detection with ByteTrack for multi-object tracking"
    )
    
    # Confidence threshold for YOLO
    if use_yolo:
        conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum confidence for object detection"
        )
    
    # Video input options
    st.sidebar.subheader("Video Input")
    video_option = st.sidebar.radio(
        "Choose Video Source",
        ["Upload Video", "Use Sample Video", "Use Test Video"]
    )
    
    # Create sample videos
    sample_video_path, test_video_path = create_sample_videos()
    
    # Video handling
    video_file = None
    if video_option == "Upload Video":
        video_file = st.sidebar.file_uploader(
            "Upload Video File",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for tracking"
        )
    elif video_option == "Use Sample Video":
        video_file = sample_video_path
    else:  # Use Test Video
        video_file = test_video_path
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Tracking")
        
        if video_file is not None:
            # Reset tracker when video changes
            if st.session_state.video_file != video_file:
                st.session_state.tracker = None
                st.session_state.video_file = video_file
            
            # Process video
            if isinstance(video_file, str):
                cap = cv2.VideoCapture(video_file)
            else:
                # Handle uploaded file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)
            
            # Initialize tracker if needed
            if st.session_state.tracker is None and not use_yolo:
                success, frame = cap.read()
                if success:
                    # For traditional trackers, we need ROI selection
                    st.info("Please select the object to track in the video player below")
                    
                    # Display first frame for ROI selection
                    st.image(frame, channels="BGR", caption="Select object to track")
                    
                    # Manual ROI input
                    col_x, col_y, col_w, col_h = st.columns(4)
                    with col_x:
                        roi_x = st.number_input("X", min_value=0, max_value=frame.shape[1], value=100)
                    with col_y:
                        roi_y = st.number_input("Y", min_value=0, max_value=frame.shape[0], value=100)
                    with col_w:
                        roi_w = st.number_input("Width", min_value=10, max_value=frame.shape[1], value=100)
                    with col_h:
                        roi_h = st.number_input("Height", min_value=10, max_value=frame.shape[0], value=100)
                    
                    if st.button("Initialize Tracker"):
                        tracker_enum = TrackerType(tracker_type)
                        st.session_state.tracker = ObjectTracker(tracker_enum)
                        bbox = (roi_x, roi_y, roi_w, roi_h)
                        
                        if st.session_state.tracker.initialize(frame, bbox):
                            st.success("Tracker initialized successfully!")
                        else:
                            st.error("Failed to initialize tracker")
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Video processing controls
            col_play, col_reset = st.columns(2)
            
            with col_play:
                if st.button("▶️ Start Tracking"):
                    st.session_state.tracking_active = True
            
            with col_reset:
                if st.button("🔄 Reset"):
                    st.session_state.tracker = None
                    st.session_state.tracking_results = []
                    st.session_state.tracking_active = False
            
            # Process and display video
            if hasattr(st.session_state, 'tracking_active') and st.session_state.tracking_active:
                frame_placeholder = st.empty()
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Process frame
                    processed_frame = process_video_frame(frame, tracker_type, use_yolo)
                    
                    # Display frame
                    frame_placeholder.image(processed_frame, channels="BGR")
                    
                    # Add small delay
                    import time
                    time.sleep(0.03)  # ~30 FPS
                    
                    # Check for stop button
                    if st.button("⏹️ Stop", key="stop_button"):
                        break
            
            cap.release()
            if isinstance(video_file, str) == False:
                os.unlink(tfile.name)
        
        else:
            st.info("Please select a video source from the sidebar")
    
    with col2:
        st.subheader("Tracking Information")
        
        # Metrics
        if st.session_state.tracker is not None:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Tracker Type", tracker_type)
            st.metric("Status", "Active" if st.session_state.tracker.is_initialized else "Inactive")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tracking history
        if st.session_state.tracking_results:
            st.subheader("Tracking History")
            for i, result in enumerate(st.session_state.tracking_results[-10:]):
                st.write(f"Frame {i}: {'Success' if result.success else 'Lost'}")
        
        # Instructions
        st.subheader("Instructions")
        st.markdown("""
        1. **Select Tracker**: Choose from various tracking algorithms
        2. **Upload Video**: Upload your own video or use sample videos
        3. **Initialize**: For traditional trackers, select the object to track
        4. **Start Tracking**: Begin real-time object tracking
        5. **Monitor**: Watch the tracking results and metrics
        """)
        
        # Algorithm descriptions
        st.subheader("Algorithm Descriptions")
        algorithm_info = {
            "CSRT": "High accuracy, good with occlusion",
            "KCF": "Fast and efficient",
            "MOSSE": "Very fast, good for real-time",
            "MIL": "Robust to appearance changes",
            "BOOSTING": "Classic algorithm, slower",
            "TLD": "Learning-based tracker",
            "MEDIANFLOW": "Good for predictable motion",
            "GOTURN": "Deep learning-based"
        }
        
        if tracker_type in algorithm_info:
            st.info(f"**{tracker_type}**: {algorithm_info[tracker_type]}")

if __name__ == "__main__":
    main()
