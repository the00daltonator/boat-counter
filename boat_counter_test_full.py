# === Boat Counter Test Full Version with Sheets + Mask ===
#
# ✅ Detects boats from a test video
# ✅ Tracks and counts them with SORT
# ✅ Supports region masking with "mask.png"
# ✅ Saves snapshots of detected boats
# ✅ Logs to Google Sheets if creds and internet available

import cv2
import numpy as np
import math
import time
import os
from datetime import datetime
from ultralytics import YOLO
from sort import *  # SORT = Simple Online and Realtime Tracking
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === CONFIGURATION ===
VIDEO_SOURCE = "test_boats.mp4"  # Path to your boat video
MODEL_PATH = "yolov8n.pt"  # Path to the YOLOv8 model
CLASS_FILTER = "boat"  # Only count boats (could change to person, car, etc.)
CONFIDENCE_THRESHOLD = 0.3  # Minimum detection confidence
COUNT_LINE = [640 // 2, 150, 640 // 2, 350]  # Vertical count line coords
SNAPSHOT_DIR = "snapshots"  # Folder for saved images
GSHEET_CREDS_FILE = "gsheets_creds.json"  # Service account JSON file
GOOGLE_SHEET_NAME = "Boat Counter Logs"  # Name of Google Sheet

# === GOOGLE SHEETS SETUP ===
sheet = None
try:
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GSHEET_CREDS_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open(GOOGLE_SHEET_NAME).sheet1
    print("[DEBUG] Connected to Google Sheets")
except Exception as e:
    print(f"[WARN] Google Sheets not connected: {e}")

# === MASK LOADING ===
mask = None
if os.path.exists("mask.png"):
    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    print("[DEBUG] Mask loaded")
else:
    print("[DEBUG] No mask found – full frame used")

# === SNAPSHOT FOLDER ===
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# === VIDEO AND MODEL INITIALIZATION ===
print("[DEBUG] Loading video and model...")
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
model = YOLO(MODEL_PATH)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []
last_time = time.time()

print("[DEBUG] Starting test...")

# === MAIN LOOP ===
while True:
    success, img = cap.read()
    if not success:
        print("[DEBUG] End of video or failed frame.")
        break

    # === MASKING ===
    if mask is not None:
        imgMasked = cv2.bitwise_and(img, img, mask=mask)
    else:
        imgMasked = img

    # === DETECTION ===
    detections = np.empty((0, 5))
    results = model(imgMasked, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = model.names[cls]
            if currentClass == CLASS_FILTER and conf > CONFIDENCE_THRESHOLD:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # === TRACKING ===
    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # === COUNTING ===
        if COUNT_LINE[1] < cy < COUNT_LINE[3] and COUNT_LINE[0] - 15 < cx < COUNT_LINE[0] + 15:
            if id not in totalCount:
                totalCount.append(id)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                print(f"[DEBUG] Boat #{int(id)} counted at {timestamp}")

                # === SNAPSHOT ===
                filename = f"boat_{timestamp}.jpg"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                cv2.imwrite(filepath, img)

                # === SHEETS LOG ===
                if sheet:
                    now = datetime.now()
                    try:
                        sheet.append_row([
                            now.strftime('%Y-%m-%d'),
                            now.strftime('%H:%M:%S'),
                            len(totalCount),
                            filename
                        ])
                    except Exception as e:
                        print(f"[ERROR] Google Sheets logging failed: {e}")

        # === VISUAL DEBUG ===
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, f"ID: {int(id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)

    # === LINE & COUNT ===
    cv2.line(img, (COUNT_LINE[0], COUNT_LINE[1]), (COUNT_LINE[2], COUNT_LINE[3]), (0, 0, 255), 2)
    cv2.putText(img, f'Total: {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # === DISPLAY FRAME ===
    cv2.imshow("Boat Detection Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
print(f"[DEBUG] Final count: {len(totalCount)} boats")
