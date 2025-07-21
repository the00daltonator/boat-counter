# Boat Counter

A computer vision application that detects and counts boats in video streams using YOLOv8 and SORT (Simple Online and Realtime Tracking).

## Features

- Real-time boat detection using YOLOv8
- Object tracking with SORT algorithm
- Line crossing detection for counting
- Visual debugging with bounding boxes and tracking IDs
- Optional Google Sheets integration for logging
- Optional region-of-interest masking

## Setup & Usage

The project uses a Makefile for all operations. After cloning the repository:

```bash
# Setup environment and install all dependencies
make setup

# Run the application with default settings
make run

# Run all tests
make test

# Run tests with coverage report
make coverage

# Clean up build artifacts and virtual environment
make clean
```

The first run will automatically download the YOLOv8 model.

## Command Line Options

When not using the Makefile, you can run directly with options:

```bash
python boat_counter.py --source video.mp4 --display
```

Common options:
- `--source`: Video source (0=webcam, path to video file, rtsp URL)
- `--display`: Enable visual display
- `--confidence`: Detection confidence threshold (0.0-1.0)
- `--line`: Counting line coordinates [x1 y1 x2 y2]

## Project Structure

```
boat-counter/
├── boat_counter.py    # Main application module
├── sort.py            # SORT tracking algorithm
├── requirements.txt   # Dependencies
├── Makefile           # Build automation
└── tests/             # Test suite
```

## Troubleshooting

If you encounter issues:

1. **Dependencies**: Run `make clean` followed by `make setup` to reset the environment
2. **Video source**: Ensure video files exist or webcam is connected
3. **Model download**: The model downloads automatically, but can be manually downloaded from Ultralytics

## Advanced: Google Sheets Integration

To enable Google Sheets logging:

1. Obtain a Google API service account credentials file
2. Run with: `python boat_counter.py --gsheets --gsheets-creds credentials.json` 