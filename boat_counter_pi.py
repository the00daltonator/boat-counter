import cv2
import numpy as np
import math
import time
from ultralytics import YOLO
from sort import *  # Make sure SORT is properly installed and available

# === CONFIG ===
VIDEO_SOURCE = 0  # Change to 0 for webcam or "video.mp4" for local file
MODEL_PATH = "yolov8n.pt"  # Use nano model for Raspberry Pi
CLASS_FILTER = "boat"
CONFIDENCE_THRESHOLD = 0.3

# === LINE SETTINGS ===
LIMITS = [100, 300, 500, 300]  # Virtual line coords (x1, y1, x2, y2)
totalCount = []  # List of unique IDs

# === INITIALIZE ===
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Load binary mask
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
if mask is not None:
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]  # Ensure binary

last_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    # Apply mask if available
    if mask is not None:
        imgMasked = cv2.bitwise_and(img, img, mask=mask)
    else:
        imgMasked = img

    # Throttle YOLO to 1 frame/sec
    if time.time() - last_time >= 1:
        detections = np.empty((0, 5))

        results = model(imgMasked, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
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

            # Check if boat crosses virtual line
            if LIMITS[0] < cx < LIMITS[2] and LIMITS[1] - 15 < cy < LIMITS[1] + 15:
                if id not in totalCount:
                    totalCount.append(id)
                    print(f"[{time.strftime('%H:%M:%S')}] Boat #{int(id)} counted.")

        print(f"Total Boats: {len(totalCount)}")
        last_time = time.time()

    # === DEBUG VIEW (optional) ===
    # cv2.line(img, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), (0, 0, 255), 2)  # Show line
    # cv2.putText(img, f'Count: {len(totalCount)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # cv2.imshow("Masked Frame", imgMasked)
    # cv2.imshow("Original Frame", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
