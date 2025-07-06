# Boat Counter

A computer vision application that detects and counts boats in video streams using YOLOv8 and SORT (Simple Online and Realtime Tracking).

## Features

- Real-time boat detection using YOLOv8
- Object tracking with SORT algorithm
- Line crossing detection for counting
- Visual debugging with bounding boxes and tracking IDs
- Support for video file processing

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics (YOLOv8)
- NumPy

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd boat-counter
```

2. Install dependencies:
```bash
pip install ultralytics opencv-python numpy
```

3. Download the YOLOv8 model:
```bash
# The yolov8n.pt file should be downloaded automatically by ultralytics
# or you can download it manually from the Ultralytics repository
```

## Usage

1. Place your video file in the project directory
2. Update the `VIDEO_SOURCE` variable in `boat_counter_video_test.py` to point to your video file
3. Adjust the `COUNT_LINE` coordinates to match your video's perspective
4. Run the script:
```bash
python boat_counter_video_test.py
```

## Configuration

Key parameters in `boat_counter_video_test.py`:

- `VIDEO_SOURCE`: Path to your video file
- `MODEL_PATH`: Path to YOLO model file
- `CLASS_FILTER`: Object class to detect ("boat")
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detection (0.3)
- `COUNT_LINE`: Line coordinates for counting [x1, y1, x2, y2]

## Files

- `boat_counter_video_test.py`: Main boat detection and counting script
- `sort.py`: SORT tracking algorithm implementation
- `test_boats.mp4`: Sample video file (not included in repo due to size)
- `yolov8n.pt`: YOLOv8 model file (not included in repo due to size)

## Notes

- Large files (video files and model files) are excluded from this repository
- You'll need to download the YOLOv8 model separately or let ultralytics handle it automatically
- The SORT tracker helps maintain consistent object IDs across frames

## License

[Add your license here] 