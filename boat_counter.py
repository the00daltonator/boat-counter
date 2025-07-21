#!/usr/bin/env python3
"""
Boat Counter
===========

A unified boat detection and counting system that works across platforms,
with optimizations for Raspberry Pi. This system detects and counts boats
crossing a virtual line in video streams.

Features:
- Real-time boat detection using YOLOv8
- Object tracking with SORT algorithm
- Line crossing detection for counting
- Optional region-of-interest masking
- Optional Google Sheets logging
- Optional snapshot capture
- Performance optimizations for resource-constrained devices
"""

import argparse
import cv2
import math
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Failed to import ultralytics. Install with 'pip install ultralytics'")
    exit(1)

from sort import Sort

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

try:
    import socket
    SOCKET_AVAILABLE = True
except ImportError:
    SOCKET_AVAILABLE = False


class BoatCounter:
    """Main class for boat detection and counting."""
    
    def __init__(self, config: Dict = None):
        """Initialize the boat counter with the provided configuration."""
        # Default configuration
        self.config = {
            # Video source (0=webcam, path for file, rtsp:// for IP camera)
            'video_source': 0,
            # YOLOv8 model path
            'model_path': 'yolov8n.pt',
            # Object class to detect
            'class_filter': 'boat',
            # Minimum detection confidence (0-1)
            'confidence_threshold': 0.3,
            # Count line coordinates [x1, y1, x2, y2]
            'count_line': [100, 300, 500, 300],
            # Directory to save detection snapshots
            'snapshot_dir': 'snapshots',
            # Google Sheets settings
            'use_gsheets': False,
            'gsheets_name': 'Boat Counter Logs',
            'gsheets_creds_file': 'gsheets_creds.json',
            # Binary mask image path (None for no mask)
            'mask_path': None,
            # Frame processing rate (seconds)
            'frame_rate': 1.0,
            # Enable visual display
            'enable_display': False,
            # SORT tracker parameters
            'max_age': 20,
            'min_hits': 3,
            'iou_threshold': 0.3,
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
            
        # Initialize counters and storage
        self.total_count = []
        self.last_frame_time = time.time()
        
        # State tracking
        self.running = False
        self.sheet = None
        self.cap = None
        self.model = None
        self.tracker = None
        self.mask = None
        
    def initialize(self) -> bool:
        """Set up camera, model, tracker, and other components."""
        # Set up video capture
        print(f"[INFO] Initializing video source: {self.config['video_source']}")
        self.cap = cv2.VideoCapture(self.config['video_source'])
        
        if not self.cap.isOpened():
            print("[ERROR] Failed to open video source")
            return False
            
        # Set preferred resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        
        # Load YOLO model
        print(f"[INFO] Loading YOLOv8 model: {self.config['model_path']}")
        try:
            self.model = YOLO(self.config['model_path'])
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            return False
            
        # Initialize SORT tracker
        self.tracker = Sort(
            max_age=self.config['max_age'],
            min_hits=self.config['min_hits'],
            iou_threshold=self.config['iou_threshold']
        )
        
        # Load mask if specified
        if self.config['mask_path']:
            print(f"[INFO] Loading mask: {self.config['mask_path']}")
            self.mask = cv2.imread(self.config['mask_path'], cv2.IMREAD_GRAYSCALE)
            if self.mask is not None:
                # Ensure mask is binary
                self.mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)[1]
                print("[INFO] Mask loaded successfully")
            else:
                print("[WARN] Failed to load mask - will process entire frame")
                
        # Set up Google Sheets integration if enabled
        if self.config['use_gsheets']:
            if not GSHEETS_AVAILABLE:
                print("[WARN] Google Sheets integration unavailable - missing dependencies")
            else:
                self._setup_gsheets()
                
        # Create snapshot directory if needed
        if self.config['snapshot_dir']:
            os.makedirs(self.config['snapshot_dir'], exist_ok=True)
            
        # Ready to run
        self.running = True
        return True
        
    def _setup_gsheets(self) -> None:
        """Initialize Google Sheets connection."""
        if not self._check_internet():
            print("[WARN] No internet connection - Google Sheets disabled")
            return
            
        try:
            print("[INFO] Connecting to Google Sheets...")
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            
            creds_file = self.config['gsheets_creds_file']
            if not os.path.exists(creds_file):
                print(f"[WARN] Credentials file not found: {creds_file}")
                return
                
            creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
            client = gspread.authorize(creds)
            self.sheet = client.open(self.config['gsheets_name']).sheet1
            print("[INFO] Google Sheets connected")
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to Google Sheets: {e}")
            
    def _check_internet(self) -> bool:
        """Check for internet connectivity."""
        if not SOCKET_AVAILABLE:
            return False
            
        try:
            # Try to connect to Google's DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """Process a single frame for boat detection and counting."""
        # Apply mask if available
        if self.mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        else:
            masked_frame = frame
            
        # Initialize detections array
        detections = np.empty((0, 5))
        
        # Run YOLO detection
        results = self.model(masked_frame, stream=True)
        
        # Process results
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Extract confidence and class
                # Fix deprecated array to scalar conversion
                conf_array = box.conf[0]
                conf = float(conf_array.item() if hasattr(conf_array, 'item') else conf_array)
                
                cls_array = box.cls[0]
                cls = int(cls_array.item() if hasattr(cls_array, 'item') else cls_array)
                
                class_name = self.model.names[cls]
                
                # Filter by class and confidence
                if class_name == self.config['class_filter'] and conf > self.config['confidence_threshold']:
                    # Add to detections
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))
        
        # Update tracker
        results_tracker = self.tracker.update(detections)
        
        # Process tracking results for line crossing
        new_detections = []
        
        for result in results_tracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            
            # Check if object crosses the counting line
            line = self.config['count_line']
            if self._check_line_crossing(cx, cy, line):
                if int(id) not in self.total_count:
                    self.total_count.append(int(id))
                    new_detections.append((int(id), cx, cy, frame))
            
            # Draw visuals if display is enabled
            if self.config['enable_display']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"ID: {int(id)}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        
        # Draw count line if display is enabled
        if self.config['enable_display']:
            cv2.line(frame, 
                    (self.config['count_line'][0], self.config['count_line'][1]), 
                    (self.config['count_line'][2], self.config['count_line'][3]), 
                    (0, 0, 255), 2)
            cv2.putText(frame, f'Total: {len(self.total_count)}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, new_detections
    
    def _check_line_crossing(self, cx: int, cy: int, line: List[int]) -> bool:
        """Check if a point crosses the counting line with tolerance."""
        # Horizontal line check
        if line[1] == line[3]:
            # Check if point is within horizontal range and near the y-coordinate
            if line[0] < cx < line[2] and line[1] - 15 < cy < line[1] + 15:
                return True
        # Vertical line check
        elif line[0] == line[2]:
            # Check if point is within vertical range and near the x-coordinate
            if line[1] < cy < line[3] and line[0] - 15 < cx < line[0] + 15:
                return True
        # Diagonal line - more complex, not implemented yet
        return False
    
    def handle_new_detections(self, detections: List) -> None:
        """Handle newly detected boats (logging, snapshots)."""
        for id, cx, cy, frame in detections:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Log detection
            print(f"[{timestamp}] Boat #{id} counted")
            
            # Save snapshot if enabled
            if self.config['snapshot_dir']:
                img_path = os.path.join(self.config['snapshot_dir'], f"boat_{timestamp}_{id}.jpg")
                cv2.imwrite(img_path, frame)
            
            # Log to Google Sheets if enabled
            if self.sheet:
                try:
                    now = datetime.now()
                    self.sheet.append_row([
                        now.strftime('%Y-%m-%d'),
                        now.strftime('%H:%M:%S'),
                        len(self.total_count),
                        f"boat_{timestamp}_{id}.jpg"
                    ])
                except Exception as e:
                    print(f"[ERROR] Failed to log to Google Sheets: {e}")
    
    def run(self) -> None:
        """Run the boat counter main loop."""
        if not self.running:
            if not self.initialize():
                print("[ERROR] Failed to initialize boat counter")
                return
        
        print("[INFO] Starting boat detection...")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame")
                    break
                
                # Process frame at specified rate
                current_time = time.time()
                if current_time - self.last_frame_time >= self.config['frame_rate']:
                    # Process the frame
                    processed_frame, new_detections = self.process_frame(frame)
                    
                    # Handle any new detections
                    if new_detections:
                        self.handle_new_detections(new_detections)
                    
                    # Update timing
                    self.last_frame_time = current_time
                    
                    # Display status
                    print(f"[INFO] Total boats counted: {len(self.total_count)}")
                
                # Show video if display is enabled
                if self.config['enable_display']:
                    cv2.imshow("Boat Counter", frame)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] User requested exit")
                        break
                        
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user")
        finally:
            # Clean up
            if self.cap:
                self.cap.release()
            if self.config['enable_display']:
                cv2.destroyAllWindows()
            
            print(f"[INFO] Final count: {len(self.total_count)} boats")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Boat Counter")
    
    parser.add_argument("-s", "--source", type=str, default="0",
                        help="Video source (0 for webcam, file path, or RTSP URL)")
    parser.add_argument("-m", "--model", type=str, default="yolov8n.pt",
                        help="Path to YOLOv8 model")
    parser.add_argument("-c", "--confidence", type=float, default=0.3,
                        help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--class", type=str, default="boat", dest="class_filter",
                        help="Class to detect (default: boat)")
    parser.add_argument("--line", type=int, nargs=4, default=[100, 300, 500, 300],
                        help="Counting line coordinates: x1 y1 x2 y2")
    parser.add_argument("--mask", type=str, default=None,
                        help="Path to binary mask image")
    parser.add_argument("--snapshots", type=str, default="snapshots",
                        help="Directory to save detection snapshots")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Frame processing rate in seconds (default: 1.0)")
    parser.add_argument("--display", action="store_true",
                        help="Enable visual display")
    parser.add_argument("--gsheets", action="store_true",
                        help="Enable Google Sheets logging")
    parser.add_argument("--gsheets-name", type=str, default="Boat Counter Logs",
                        help="Google Sheets name")
    parser.add_argument("--gsheets-creds", type=str, default="gsheets_creds.json",
                        help="Path to Google Sheets credentials file")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert numeric source if needed
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Create config from args
    config = {
        'video_source': source,
        'model_path': args.model,
        'class_filter': args.class_filter,
        'confidence_threshold': args.confidence,
        'count_line': args.line,
        'snapshot_dir': args.snapshots,
        'use_gsheets': args.gsheets,
        'gsheets_name': args.gsheets_name,
        'gsheets_creds_file': args.gsheets_creds,
        'mask_path': args.mask,
        'frame_rate': args.rate,
        'enable_display': args.display,
    }
    
    # Create and run boat counter
    counter = BoatCounter(config)
    counter.run()


if __name__ == "__main__":
    main() 