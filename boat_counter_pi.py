"""
Boat Counter for Raspberry Pi
============================

This script is optimized for Raspberry Pi deployment with the following considerations:
- Reduced frame rate (1 FPS) to conserve CPU resources
- Smaller YOLO model (yolov8n) for faster inference
- Optional binary mask for region-of-interest filtering
- Minimal visual output to reduce overhead
- Memory-efficient tracking with SORT algorithm

Author: [Your Name]
Date: [Date]
"""

import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
from sort import *  # SORT (Simple Online and Realtime Tracking) for object tracking

# === CONFIGURATION SECTION ===
# These settings can be adjusted based on your specific deployment environment

VIDEO_SOURCE = 0  # 0 = USB webcam, "video.mp4" = local video file, "rtsp://..." = IP camera
MODEL_PATH = "yolov8n.pt"  # YOLOv8 nano model - smallest and fastest for Raspberry Pi
CLASS_FILTER = "boat"  # Only detect boats (can be changed to "person", "car", etc.)
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence (0.0-1.0) - higher = fewer false positives

# === LINE CROSSING DETECTION SETTINGS ===
# Virtual line coordinates: [x1, y1, x2, y2] - boats crossing this line will be counted
# Adjust these coordinates based on your camera angle and desired counting zone
LIMITS = [100, 300, 500, 300]  # Horizontal line at y=300, from x=100 to x=500

# Storage for unique boat IDs that have crossed the line
# Using a list to prevent double-counting the same boat
totalCount = []

# === HARDWARE INITIALIZATION ===
# Initialize video capture - this connects to your camera or video source
print("[INFO] Initializing video capture...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Set frame resolution - lower resolution = faster processing
# 640x360 is a good balance between performance and detection accuracy
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# === AI MODEL INITIALIZATION ===
# Load YOLO model for object detection
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Initialize SORT tracker for maintaining object IDs across frames
# max_age: How many frames to keep tracking an object after it disappears
# min_hits: How many consecutive detections before assigning a track ID
# iou_threshold: Intersection over Union threshold for track association
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# === OPTIONAL: BINARY MASK FOR REGION OF INTEREST ===
# A binary mask can be used to focus detection on specific areas
# This is useful for ignoring irrelevant parts of the frame (e.g., sky, land)
print("[INFO] Loading binary mask (if available)...")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
if mask is not None:
    # Ensure mask is binary (0 or 255) for proper masking
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    print("[INFO] Binary mask loaded successfully")
else:
    print("[INFO] No mask found - processing entire frame")

# Performance tracking - used for frame rate limiting
last_time = time.time()

print("[INFO] Starting boat detection loop...")

# === MAIN PROCESSING LOOP ===
while True:
    # Capture frame from video source
    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to read frame - check video source")
        break

    # === APPLY BINARY MASK (IF AVAILABLE) ===
    # Masking helps focus detection on water areas and ignore irrelevant regions
    # This improves both accuracy and performance
    if mask is not None:
        # Apply mask to focus on region of interest
        imgMasked = cv2.bitwise_and(img, img, mask=mask)
    else:
        # Process entire frame if no mask is available
        imgMasked = img

    # === FRAME RATE LIMITING FOR RASPBERRY PI ===
    # YOLO inference is computationally expensive
    # Limiting to 1 FPS conserves CPU resources and prevents overheating
    if time.time() - last_time >= 1:  # Process only once per second
        
        # Initialize empty array for detections
        # Format: [x1, y1, x2, y2, confidence] for each detected object
        detections = np.empty((0, 5))

        # === YOLO OBJECT DETECTION ===
        # Run YOLO model on the masked image
        # stream=True enables memory-efficient processing
        results = model(imgMasked, stream=True)
        
        # Process each detection result
        for r in results:
            boxes = r.boxes  # Get bounding boxes from YOLO output
            
            # Process each detected object
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right corners
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1  # Calculate width and height

                # Extract confidence score and class
                conf = math.ceil((box.conf[0] * 100)) / 100  # Round to 2 decimal places
                cls = int(box.cls[0])  # Class index
                currentClass = model.names[cls]  # Convert index to class name

                # Filter detections: only boats with sufficient confidence
                if currentClass == CLASS_FILTER and conf > CONFIDENCE_THRESHOLD:
                    # Add detection to array for tracking
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        # === OBJECT TRACKING WITH SORT ===
        # Update tracker with new detections
        # Returns: [x1, y1, x2, y2, track_id] for each tracked object
        resultsTracker = tracker.update(detections)

        # === LINE CROSSING DETECTION ===
        # Process each tracked object
        for result in resultsTracker:
            x1, y1, x2, y2, id = result  # Extract tracking result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Calculate center point of the bounding box
            cx, cy = x1 + w // 2, y1 + h // 2

            # Check if boat center crosses the virtual counting line
            # Line is defined by LIMITS: [x1, y1, x2, y2]
            # Tolerance of ±15 pixels accounts for object size and tracking jitter
            if LIMITS[0] < cx < LIMITS[2] and LIMITS[1] - 15 < cy < LIMITS[1] + 15:
                # Only count if this boat ID hasn't been counted before
                if id not in totalCount:
                    totalCount.append(id)
                    print(f"[{time.strftime('%H:%M:%S')}] Boat #{int(id)} counted.")

        # Display current count
        print(f"Total Boats: {len(totalCount)}")
        
        # Update timestamp for next frame processing
        last_time = time.time()

    # === OPTIONAL DEBUG VISUALIZATION ===
    # Uncomment these lines for visual debugging (may impact performance)
    # cv2.line(img, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), (0, 0, 255), 2)  # Show counting line
    # cv2.putText(img, f'Count: {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)  # Show count
    # cv2.imshow("Masked Frame", imgMasked)  # Show masked frame
    # cv2.imshow("Original Frame", img)  # Show original frame

    # === EXIT CONDITION ===
    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] User requested exit")
        break

# === CLEANUP ===
# Release resources to prevent memory leaks
print("[INFO] Cleaning up resources...")
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Final boat count: {len(totalCount)}")
print("[INFO] Boat counter stopped successfully")
