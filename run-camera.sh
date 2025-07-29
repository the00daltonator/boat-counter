#!/bin/bash

# Script to run boat counter with camera access
# This script handles the differences between macOS and Linux

echo "Starting Boat Counter with camera access..."

# Check if we're on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS..."
    echo "Note: Camera access in Docker on macOS has limitations."
    echo "For best results, run the application locally: make run"
    echo ""
    echo "Attempting Docker camera access (may not work)..."
    
    # On macOS, camera access in Docker is limited
    # We'll try using the host network approach
    docker run --rm \
        --network host \
        --privileged \
        -e DISPLAY=host.docker.internal:0 \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd)/snapshots:/app/snapshots \
        -v $(pwd)/videos:/app/videos:ro \
        boat-counter \
        python boat_counter.py --source 0 --display
        
    echo ""
    echo "If camera didn't work, try running locally with: make run"
    
else
    echo "Detected Linux - using standard camera setup..."
    
    # On Linux, we can directly mount the video device
    docker run --rm \
        --network host \
        --privileged \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd)/snapshots:/app/snapshots \
        -v $(pwd)/videos:/app/videos:ro \
        --device /dev/video0:/dev/video0 \
        boat-counter \
        python boat_counter.py --source 0 --display
fi 