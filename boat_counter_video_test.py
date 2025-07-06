# === Boat Detection Test with Video File ===

import cv2
import numpy as np
import math
import time
import os
from datetime import datetime
from ultralytics import YOLO
from sort import *  # SORT Tracker

# === CONFIGURATION ===
VIDEO_SOURCE = "test_boats.mp4"  # Path to your boat video
MODEL_PATH = "yolov8n.pt"
CLASS_FILTER = "boat"
CONFIDENCE_THRESHOLD = 0.3
COUNT_LINE = [100, 300, 500, 300]  # Adjust based on your video frame

# === INITIALIZATION ===
print("[DEBUG] Loading video...")
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("[DEBUG] Loading YOLO model...")
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

    # Run YOLO once per frame for testing
    detections = np.empty((0, 5))
    results = model(img, stream=True)

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

        # Check crossing line
        if COUNT_LINE[0] < cx < COUNT_LINE[2] and COUNT_LINE[1] - 15 < cy < COUNT_LINE[1] + 15:
            if id not in totalCount:
                totalCount.append(id)
                print(f"[DEBUG] Boat #{int(id)} counted at {datetime.now().strftime('%H:%M:%S')}")

        # === VISUAL DEBUG ===
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, f"ID: {int(id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)

    # Show count line and total
    cv2.line(img, (COUNT_LINE[0], COUNT_LINE[1]), (COUNT_LINE[2], COUNT_LINE[3]), (0, 0, 255), 2)
    cv2.putText(img, f'Total: {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Boat Detection Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
cv2.destroyAllWindows()
print(f"[DEBUG] Final count: {len(totalCount)} boats")
