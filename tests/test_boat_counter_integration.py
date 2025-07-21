#!/usr/bin/env python3
"""
Integration tests for Boat Counter
=================================

This module contains integration tests for the boat_counter.py module,
testing the complete detection and counting pipeline with sample data.
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import the boat_counter module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boat_counter import BoatCounter

# Skip integration tests if running in CI environment
SKIP_INTEGRATION_TESTS = os.environ.get("CI", "false").lower() == "true"


@unittest.skipIf(SKIP_INTEGRATION_TESTS, "Skipping integration tests in CI environment")
class TestBoatCounterIntegration(unittest.TestCase):
    """Integration test cases for the BoatCounter class."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources once for all tests."""
        # Create test data directory if it doesn't exist
        cls.test_data_dir = Path(__file__).parent / 'data'
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # Check if test video exists, if not, create a dummy video
        cls.test_video_path = cls.test_data_dir / 'test_video.mp4'
        if not cls.test_video_path.exists():
            cls._create_test_video()

    @classmethod
    def _create_test_video(cls):
        """Create a dummy test video with a moving rectangle."""
        import cv2
        
        # Video settings
        width, height = 640, 360
        fps = 30
        duration = 3  # seconds
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(cls.test_video_path), fourcc, fps, (width, height))
        
        # Generate frames with a moving rectangle
        for i in range(fps * duration):
            # Create a blank frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate rectangle position (moving from left to right)
            x_pos = int((i / (fps * duration)) * width)
            
            # Draw rectangle (simulating a boat)
            cv2.rectangle(frame, (x_pos, height//2-30), (x_pos+100, height//2+30), (0, 0, 255), -1)
            
            # Write frame to video
            out.write(frame)
            
        # Release resources
        out.release()
        
        print(f"Created test video at {cls.test_video_path}")
    
    def setUp(self):
        """Set up test environment before each test case."""
        # Configure test configuration
        self.test_config = {
            'video_source': str(self.test_video_path),
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.3,
            'enable_display': False,
            'snapshot_dir': None,  # Disable snapshots for tests
            'use_gsheets': False,  # Disable Google Sheets for tests
            'count_line': [150, 180, 250, 180],  # Line that will intersect with our mock detection
        }
        
    def test_detection_pipeline_with_mock_model(self):
        """Test the complete detection pipeline with a mock model."""
        # This test uses the real pipeline but with mocked detection results
        import cv2
        from unittest.mock import MagicMock, patch
        
        # Open the test video
        cap = cv2.VideoCapture(str(self.test_video_path))
        
        # Get the first frame for testing
        ret, frame = cap.read()
        self.assertTrue(ret, "Failed to read test video frame")
        
        # Clean up
        cap.release()
        
        # Create counter with mocked components
        counter = BoatCounter(self.test_config)
        
        # Mock the model to always return a boat detection
        counter.model = MagicMock()
        mock_results = MagicMock()
        mock_boxes = MagicMock()
        
        # Set up the detection result (simulated boat)
        mock_box = MagicMock()
        mock_box.xyxy = [np.array([100, 150, 200, 250])]
        mock_box.conf = [np.array([0.8])]
        mock_box.cls = [np.array([0])]
        
        mock_boxes.boxes = [mock_box]
        mock_results.__iter__.return_value = [mock_boxes]
        counter.model.return_value = mock_results
        counter.model.names = {0: "boat"}
        
        # Create mock tracker that returns a track crossing our counting line
        counter.tracker = MagicMock()
        # Position the "boat" to cross the count line defined in test_config
        counter.tracker.update.return_value = np.array([[150, 180, 250, 180, 1]])
        
        # Process the frame
        counter.running = True
        processed_frame, new_detections = counter.process_frame(frame)
        
        # Verify detection
        self.assertEqual(len(new_detections), 1, "Should detect one boat")
        
        if len(new_detections) > 0:  # Guard against test failure for meaningful feedback
            detected_id = new_detections[0][0]
            self.assertEqual(detected_id, 1, "Boat ID should be 1")
            
            # Verify counter
            self.assertEqual(len(counter.total_count), 1, "Should count one boat")
            self.assertEqual(counter.total_count[0], 1, "Counted boat ID should be 1")


if __name__ == '__main__':
    unittest.main() 