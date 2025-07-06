# === Boat Detection + Logging with Google Sheets + Optional LTE ===

import cv2
import numpy as np
import math
import time
import os
from datetime import datetime
from ultralytics import YOLO
from sort import *  # SORT Tracker
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import socket

# === CONFIGURATION ===
VIDEO_SOURCE = 0  # 0 = webcam; or use 'video.mp4'
MODEL_PATH = "yolov8n.pt"  # Lightweight model
CLASS_FILTER = "boat"       # Detect boats only
CONFIDENCE_THRESHOLD = 0.3  # Minimum detection confidence
SNAPSHOT_DIR = "snapshots"  # Folder for saved images
GOOGLE_SHEET_NAME = "Boat Counter Logs"
GSHEET_CREDS_FILE = "gsheets_creds.json"  # Google credentials
COUNT_LINE = [100, 300, 500, 300]  # Line coords: (x1, y1, x2, y2)

# === CONNECTIVITY CHECK ===
def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53))
        print("[DEBUG] Internet connection: OK")
        return True
    except OSError:
        print("[DEBUG] Internet connection: FAILED")
        return False

# === INITIALIZATION ===
print("[DEBUG] Initializing camera...")
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("[DEBUG] Loading YOLO model...")
model = YOLO(MODEL_PATH)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
totalCount = []
last_time = time.time()

print("[DEBUG] Checking for mask file...")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
if mask is not None:
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    print("[DEBUG] Mask loaded.")
else:
    print("[DEBUG] No mask found.")

# === GOOGLE SHEETS SETUP ===
sheet = None
if check_internet():
    try:
        print("[DEBUG] Connecting to Google Sheets...")
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(GSHEET_CREDS_FILE, scope)
        sheet = gspread.authorize(creds).open(GOOGLE_SHEET_NAME).sheet1
        print("[DEBUG] Google Sheets connected.")
    except Exception as e:
        print(f"[ERROR] Failed to connect to Google Sheets: {e}")
else:
    print("[WARN] No internet. Skipping Google Sheets upload.")

# === SNAPSHOT DIRECTORY ===
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
print("[DEBUG] Snapshot directory checked/created.")

# === MAIN LOOP ===
while True:
    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to read from camera.")
        break

    if mask is not None:
        imgMasked = cv2.bitwise_and(img, img, mask=mask)
    else:
        imgMasked = img

    if time.time() - last_time >= 1:
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

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            if COUNT_LINE[0] < cx < COUNT_LINE[2] and COUNT_LINE[1] - 15 < cy < COUNT_LINE[1] + 15:
                if id not in totalCount:
                    totalCount.append(id)

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_path = os.path.join(SNAPSHOT_DIR, f"boat_{timestamp}.jpg")
                    cv2.imwrite(img_path, img)

                    if sheet:
                        try:
                            now = datetime.now()
                            sheet.append_row([
                                now.strftime('%Y-%m-%d'),
                                now.strftime('%H:%M:%S'),
                                len(totalCount),
                                os.path.basename(img_path)
                            ])
                            print(f"[{timestamp}] Boat #{int(id)} counted and logged.")
                        except Exception as e:
                            print(f"[ERROR] Google Sheets logging failed: {e}")
                    else:
                        print(f"[{timestamp}] Boat #{int(id)} counted (not logged – offline).")

        print(f"[DEBUG] Total Boats: {len(totalCount)}")
        last_time = time.time()

    # === DEBUG VIEW (comment out when headless) ===
    # cv2.line(img, (COUNT_LINE[0], COUNT_LINE[1]), (COUNT_LINE[2], COUNT_LINE[3]), (0, 0, 255), 2)
    # cv2.putText(img, f'Total: {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # cv2.imshow("Masked Frame", imgMasked)
    # cv2.imshow("Live", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
print("[DEBUG] Releasing camera and closing windows...")
cap.release()
cv2.destroyAllWindows()
print("[DEBUG] Shutdown complete.")
