"""Video capture module for real-time video input processing.

This module provides functionality for capturing and preprocessing video input
from system cameras, with support for face detection and alignment.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Union, List, Dict
from mtcnn import MTCNN

class VideoCapture:
    """Video capture class for handling camera input."""

    def __init__(self):
        """Initialize video capture with default settings."""
        self.device_id = 0
        self.cap = None
        self.current_resolution = (1920, 1080)  # Default to 1080p for better quality
        self.last_frame_time = time.time()
        self.fps = 30.0
        self.frame_time = time.time()
        # Initialize face detector
        try:
            self.face_detector = MTCNN()
        except Exception as e:
            print(f"Warning: Could not initialize face detector: {str(e)}")
            self.face_detector = None

    def list_devices(self) -> list:
        """List available camera devices.

        Returns:
            list: List of dictionaries containing device info
        """
        available_devices = []
        # Try first 3 device IDs only to avoid excessive scanning
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_devices.append({
                    'id': str(i),
                    'name': f'Camera {i}'
                })
                cap.release()
        return available_devices if available_devices else [{'id': '0', 'name': 'Default Camera'}]

    def start(self) -> bool:
        """Start video capture.

        Returns:
            bool: True if capture started successfully, False otherwise
        """
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            print(f"Warning: Failed to open camera {self.device_id}, falling back to default camera")
            self.device_id = 0
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open any camera")
                return False

        self.set_resolution(self.current_resolution[0], self.current_resolution[1])
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video capture device.

        Returns:
            Tuple containing:
                - bool: True if frame was successfully captured
                - np.ndarray or None: The captured frame if successful, None otherwise
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.frame_time
        self.frame_time = current_time
        self.fps = 1.0 / time_diff if time_diff > 0 else 0.0

        return True, frame

    def set_resolution(self, width: int, height: int) -> bool:
        """Set the capture resolution.

        Args:
            width: Desired frame width
            height: Desired frame height

        Returns:
            bool: True if resolution was set successfully
        """
        if not self.cap:
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify if resolution was set correctly
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.current_resolution = (int(actual_width), int(actual_height))
        
        return abs(width - actual_width) < 10 and abs(height - actual_height) < 10

    def get_fps(self) -> float:
        """Get the current frames per second.

        Returns:
            float: Current FPS
        """
        return self.fps

    def release(self) -> None:
        """Release the video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def detect_face(self, frame: np.ndarray) -> Optional[dict]:
        """Detect face in the given frame using MTCNN.

        Args:
            frame: Input frame as numpy array

        Returns:
            Optional[dict]: Face detection results including landmarks,
                           or None if no face detected
        """
        if self.face_detector is None:
            return None

        results = self.face_detector.detect_faces(frame)
        return results[0] if results else None

    def get_aligned_face(self, frame: np.ndarray,
                        target_size: Tuple[int, int] = (256, 256)
                        ) -> Optional[np.ndarray]:
        """Detect, crop and align face from input frame.

        Args:
            frame: Input frame
            target_size: Desired output size for face crop

        Returns:
            Optional[np.ndarray]: Aligned face image if detected, None otherwise
        """
        face_data = self.detect_face(frame)
        if face_data is None:
            return None

        x, y, w, h = face_data['box']
        face_img = frame[y:y+h, x:x+w]
        return cv2.resize(face_img, target_size)

    def release(self) -> None:
        """Release the video capture resources."""
        if self.cap:
            self.cap.release()