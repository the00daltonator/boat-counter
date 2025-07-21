#!/usr/bin/env python3
"""
Unit tests for Boat Counter
==========================

This module contains comprehensive tests for the boat_counter.py module,
covering functionality like initialization, detection, tracking, and counting.
"""

import os
import sys
import unittest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the boat_counter module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boat_counter import BoatCounter


class TestBoatCounter(unittest.TestCase):
    """Test cases for the BoatCounter class."""

    def setUp(self):
        """Set up test environment before each test case."""
        # Configure minimal test configuration
        self.test_config = {
            'video_source': str(Path(__file__).parent / 'data' / 'test_video.mp4'),
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'enable_display': False,
            'snapshot_dir': None,  # Disable snapshots for tests
            'use_gsheets': False,  # Disable Google Sheets for tests
        }
        
        # Create test directories if they don't exist
        self.test_data_dir = Path(__file__).parent / 'data'
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Mock object setup will be handled in individual tests

    def tearDown(self):
        """Clean up after each test case."""
        # Nothing to clean up yet
        pass

    @patch('boat_counter.YOLO')
    @patch('boat_counter.cv2.VideoCapture')
    def test_initialization(self, mock_video_capture, mock_yolo):
        """Test that the BoatCounter initializes correctly."""
        # Configure mocks
        mock_video_capture.return_value.isOpened.return_value = True
        mock_yolo.return_value = MagicMock()
        
        # Create counter instance
        counter = BoatCounter(self.test_config)
        self.assertFalse(counter.running)
        
        # Initialize counter
        with patch('boat_counter.Sort') as mock_sort:
            mock_sort.return_value = MagicMock()
            result = counter.initialize()
            
        # Check initialization result
        self.assertTrue(result)
        self.assertTrue(counter.running)
        
        # Verify mocks were called correctly
        mock_video_capture.assert_called_once_with(self.test_config['video_source'])
        mock_yolo.assert_called_once_with(self.test_config['model_path'])

    @patch('boat_counter.YOLO')
    @patch('boat_counter.cv2.VideoCapture')
    def test_initialization_failure(self, mock_video_capture, mock_yolo):
        """Test handling of initialization failures."""
        # Configure mock to simulate camera failure
        mock_video_capture.return_value.isOpened.return_value = False
        
        # Create counter instance
        counter = BoatCounter(self.test_config)
        
        # Initialize counter should fail
        result = counter.initialize()
        
        # Check initialization result
        self.assertFalse(result)
        self.assertFalse(counter.running)

    def test_line_crossing_horizontal(self):
        """Test the line crossing detection for a horizontal line."""
        counter = BoatCounter(self.test_config)
        
        # Test point crossing a horizontal line
        line = [100, 300, 500, 300]  # Horizontal line at y=300
        
        # Point on the line should be detected
        self.assertTrue(counter._check_line_crossing(250, 300, line))
        
        # Point near the line should be detected (within tolerance)
        self.assertTrue(counter._check_line_crossing(250, 310, line))
        self.assertTrue(counter._check_line_crossing(250, 290, line))
        
        # Point outside the line's x-range should not be detected
        self.assertFalse(counter._check_line_crossing(50, 300, line))
        self.assertFalse(counter._check_line_crossing(600, 300, line))
        
        # Point too far from the line should not be detected
        self.assertFalse(counter._check_line_crossing(250, 330, line))
        self.assertFalse(counter._check_line_crossing(250, 270, line))

    def test_line_crossing_vertical(self):
        """Test the line crossing detection for a vertical line."""
        counter = BoatCounter(self.test_config)
        
        # Test point crossing a vertical line
        line = [300, 100, 300, 500]  # Vertical line at x=300
        
        # Point on the line should be detected
        self.assertTrue(counter._check_line_crossing(300, 250, line))
        
        # Point near the line should be detected (within tolerance)
        self.assertTrue(counter._check_line_crossing(310, 250, line))
        self.assertTrue(counter._check_line_crossing(290, 250, line))
        
        # Point outside the line's y-range should not be detected
        self.assertFalse(counter._check_line_crossing(300, 50, line))
        self.assertFalse(counter._check_line_crossing(300, 600, line))
        
        # Point too far from the line should not be detected
        self.assertFalse(counter._check_line_crossing(330, 250, line))
        self.assertFalse(counter._check_line_crossing(270, 250, line))

    @patch('boat_counter.YOLO')
    @patch('boat_counter.cv2.VideoCapture')
    @patch('boat_counter.Sort')
    def test_process_frame(self, mock_sort, mock_video_capture, mock_yolo):
        """Test frame processing."""
        # Create a test frame
        test_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        # Configure YOLO mock
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_boxes = MagicMock()
        
        # Set up the detection results
        mock_box = MagicMock()
        mock_box.xyxy = [np.array([100, 150, 200, 250])]
        mock_box.conf = [np.array([0.8])]
        mock_box.cls = [np.array([0])]
        
        mock_boxes.boxes = [mock_box]
        mock_results.__iter__.return_value = [mock_boxes]
        mock_model.return_value = mock_results
        mock_model.names = {0: "boat"}
        mock_yolo.return_value = mock_model
        
        # Configure tracker mock
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = np.array([[100, 150, 200, 250, 1]])
        mock_sort.return_value = mock_tracker
        
        # Create counter instance and manually set up components
        counter = BoatCounter(self.test_config)
        counter.model = mock_model
        counter.tracker = mock_tracker
        counter.config['count_line'] = [0, 200, 640, 200]  # Horizontal line through the middle
        
        # Process the frame
        processed_frame, new_detections = counter.process_frame(test_frame)
        
        # Check that tracker was called
        mock_tracker.update.assert_called_once()
        
        # Check that a detection was made
        self.assertEqual(len(new_detections), 1)
        
        # Check detection details
        detected_id, cx, cy, _ = new_detections[0]
        self.assertEqual(detected_id, 1)
        self.assertEqual(cx, 150)
        self.assertEqual(cy, 200)

    @patch('boat_counter.BoatCounter.initialize')
    @patch('boat_counter.BoatCounter.process_frame')
    def test_run_method(self, mock_process_frame, mock_initialize):
        """Test the run method behavior."""
        # Configure mocks
        mock_initialize.return_value = True
        mock_process_frame.return_value = (None, [])
        
        # Create counter with mock video capture
        counter = BoatCounter(self.test_config)
        counter.cap = MagicMock()
        counter.cap.read.return_value = (True, np.zeros((360, 640, 3), dtype=np.uint8))
        counter.running = True
        
        # Override run method to exit after one iteration
        original_run = counter.run
        
        def modified_run():
            counter.last_frame_time = 0  # Force processing
            # Set up to process only one frame
            counter.cap.read.side_effect = [
                (True, np.zeros((360, 640, 3), dtype=np.uint8)),
                (False, None)
            ]
            original_run()
            
        counter.run = modified_run
        
        # Run the counter
        counter.run()
        
        # Check that process_frame was called
        mock_process_frame.assert_called_once()


if __name__ == '__main__':
    unittest.main() 