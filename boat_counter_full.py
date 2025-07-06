"""
Boat Counter Full Version with Google Sheets Logging
==================================================

This is the full-featured version of the boat counter with the following capabilities:
- Real-time boat detection using YOLOv8
- Object tracking with SORT algorithm
- Line crossing detection for counting
- Automatic snapshot capture of detected boats
- Google Sheets integration for data logging
- Internet connectivity monitoring
- Optional binary mask for region-of-interest filtering
- Comprehensive error handling and logging

This version is designed for production deployment with data persistence and remote monitoring.

Author: [Your Name]
Date: [Date]
"""

# === IMPORTS ===
import cv2
import numpy as np
import math
import time
import os
from datetime import datetime
from ultralytics import YOLO
from sort import *  # SORT (Simple Online and Realtime Tracking) for object tracking
import gspread  # Google Sheets API client
from oauth2client.service_account import ServiceAccountCredentials  # Google authentication
import socket  # For internet connectivity testing

# === CONFIGURATION SECTION ===
# These settings control the behavior of the boat detection system

VIDEO_SOURCE = 0  # 0 = USB webcam, "video.mp4" = local file, "rtsp://..." = IP camera
MODEL_PATH = "yolov8n.pt"  # YOLOv8 nano model - lightweight and fast
CLASS_FILTER = "boat"  # Object class to detect (can be "person", "car", etc.)
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence (0.0-1.0) - higher = fewer false positives
SNAPSHOT_DIR = "snapshots"  # Directory to save boat detection images
GOOGLE_SHEET_NAME = "Boat Counter Logs"  # Name of your Google Sheet
GSHEET_CREDS_FILE = "gsheets_creds.json"  # Google service account credentials file
COUNT_LINE = [100, 300, 500, 300]  # Virtual line coordinates [x1, y1, x2, y2] for counting

# === INTERNET CONNECTIVITY CHECK ===
def check_internet():
    """
    Test internet connectivity by attempting to connect to Google's DNS server.
    
    Returns:
        bool: True if internet connection is available, False otherwise
    
    This function is crucial for determining whether Google Sheets logging is possible.
    It helps prevent errors when the system is deployed in areas with unreliable connectivity.
    """
    try:
        # Try to connect to Google's DNS server (8.8.8.8) on port 53
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        print("[DEBUG] Internet connection: OK")
        return True
    except OSError:
        print("[DEBUG] Internet connection: FAILED")
        return False

# === HARDWARE INITIALIZATION ===
# Set up video capture and camera settings
print("[DEBUG] Initializing camera...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Set frame resolution - 640x360 provides good balance of performance and accuracy
# Lower resolution = faster processing, higher resolution = better detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# === AI MODEL INITIALIZATION ===
# Load YOLO model for object detection
print("[DEBUG] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Initialize SORT tracker for maintaining object IDs across frames
# max_age: How many frames to keep tracking an object after it disappears
# min_hits: How many consecutive detections before assigning a track ID
# iou_threshold: Intersection over Union threshold for track association
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Storage for unique boat IDs that have crossed the counting line
# Prevents double-counting the same boat
totalCount = []

# Performance tracking for frame rate control
last_time = time.time()

# === OPTIONAL: BINARY MASK FOR REGION OF INTEREST ===
# A binary mask can focus detection on specific areas (e.g., water only)
# This improves accuracy by ignoring irrelevant regions like sky or land
print("[DEBUG] Checking for mask file...")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
if mask is not None:
    # Ensure mask is binary (0 or 255) for proper masking
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    print("[DEBUG] Mask loaded.")
else:
    print("[DEBUG] No mask found.")

# === GOOGLE SHEETS INTEGRATION SETUP ===
# Initialize Google Sheets connection for data logging
sheet = None
if check_internet():
    try:
        print("[DEBUG] Connecting to Google Sheets...")
        
        # Define the required Google API scopes
        # spreadsheets: Read/write access to Google Sheets
        # drive: Access to Google Drive for file operations
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Load service account credentials from JSON file
        # This file should be obtained from Google Cloud Console
        creds = ServiceAccountCredentials.from_json_keyfile_name(GSHEET_CREDS_FILE, scope)
        
        # Authorize and open the specified Google Sheet
        sheet = gspread.authorize(creds).open(GOOGLE_SHEET_NAME).sheet1
        print("[DEBUG] Google Sheets connected.")
        
    except Exception as e:
        print(f"[ERROR] Failed to connect to Google Sheets: {e}")
        print("[INFO] System will continue without logging capability")
else:
    print("[WARN] No internet connection. Skipping Google Sheets upload.")

# === SNAPSHOT DIRECTORY SETUP ===
# Create directory for saving boat detection images
# These snapshots provide visual evidence of detections
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
print("[DEBUG] Snapshot directory checked/created.")

print("[DEBUG] Starting boat detection and logging system...")

# === MAIN PROCESSING LOOP ===
while True:
    # Capture frame from video source
    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to read from camera.")
        break

    # === APPLY BINARY MASK (IF AVAILABLE) ===
    # Masking helps focus detection on relevant areas and ignore irrelevant regions
    # This improves both accuracy and performance
    if mask is not None:
        # Apply mask to focus on region of interest (e.g., water areas)
        imgMasked = cv2.bitwise_and(img, img, mask=mask)
    else:
        # Process entire frame if no mask is available
        imgMasked = img

    # === FRAME RATE CONTROL ===
    # Process frames at 1 FPS to balance performance and accuracy
    # This prevents overwhelming the system while maintaining good detection
    if time.time() - last_time >= 1:  # Process once per second
        
        # Initialize empty array for detections
        # Format: [x1, y1, x2, y2, confidence] for each detected object
        detections = np.empty((0, 5))
        
        # === YOLO OBJECT DETECTION ===
        # Run YOLO model on the masked image
        # stream=True enables memory-efficient processing
        results = model(imgMasked, stream=True)

        # Process each detection result from YOLO
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

        # === LINE CROSSING DETECTION AND LOGGING ===
        # Process each tracked object
        for result in resultsTracker:
            x1, y1, x2, y2, id = result  # Extract tracking result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Calculate center point of the bounding box
            cx, cy = x1 + w // 2, y1 + h // 2

            # Check if boat center crosses the virtual counting line
            # Line is defined by COUNT_LINE: [x1, y1, x2, y2]
            # Tolerance of ±15 pixels accounts for object size and tracking jitter
            if COUNT_LINE[0] < cx < COUNT_LINE[2] and COUNT_LINE[1] - 15 < cy < COUNT_LINE[1] + 15:
                # Only count if this boat ID hasn't been counted before
                if id not in totalCount:
                    totalCount.append(id)

                    # === SNAPSHOT CAPTURE ===
                    # Save image of the detected boat for visual verification
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_path = os.path.join(SNAPSHOT_DIR, f"boat_{timestamp}.jpg")
                    cv2.imwrite(img_path, img)

                    # === GOOGLE SHEETS LOGGING ===
                    # Log detection data to Google Sheets if connection is available
                    if sheet:
                        try:
                            now = datetime.now()
                            
                            # Append row to Google Sheet with detection data:
                            # - Date (YYYY-MM-DD)
                            # - Time (HH:MM:SS)
                            # - Total count (cumulative)
                            # - Image filename (for reference)
                            sheet.append_row([
                                now.strftime('%Y-%m-%d'),
                                now.strftime('%H:%M:%S'),
                                len(totalCount),
                                os.path.basename(img_path)
                            ])
                            print(f"[{timestamp}] Boat #{int(id)} counted and logged.")
                            
                        except Exception as e:
                            print(f"[ERROR] Google Sheets logging failed: {e}")
                            print("[INFO] Detection was still counted and image saved")
                    else:
                        # Log locally when Google Sheets is not available
                        print(f"[{timestamp}] Boat #{int(id)} counted (not logged – offline).")

        # Display current count for monitoring
        print(f"[DEBUG] Total Boats: {len(totalCount)}")
        
        # Update timestamp for next frame processing
        last_time = time.time()

    # === OPTIONAL DEBUG VISUALIZATION ===
    # Uncomment these lines for visual debugging (may impact performance)
    # cv2.line(img, (COUNT_LINE[0], COUNT_LINE[1]), (COUNT_LINE[2], COUNT_LINE[3]), (0, 0, 255), 2)  # Show counting line
    # cv2.putText(img, f'Total: {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)  # Show count
    # cv2.imshow("Masked Frame", imgMasked)  # Show masked frame
    # cv2.imshow("Live", img)  # Show original frame

    # === EXIT CONDITION ===
    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] User requested exit")
        break

# === CLEANUP ===
# Release resources to prevent memory leaks and ensure proper shutdown
print("[DEBUG] Releasing camera and closing windows...")
cap.release()
cv2.destroyAllWindows()
print(f"[DEBUG] Final boat count: {len(totalCount)}")
print("[DEBUG] Shutdown complete.")
