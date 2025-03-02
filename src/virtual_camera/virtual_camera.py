"""Virtual Camera implementation for video conferencing integration.

This module provides functionality to create a virtual camera output stream
that can be used in video conferencing applications, displaying the
generated avatar with synchronized lip movements.
"""

import cv2
import numpy as np
import pyvirtualcam
import threading
import time
from typing import Optional, Dict, Any, Tuple

class VirtualCamera:
    """Virtual camera class for streaming avatar output."""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """Initialize the virtual camera.
        
        Args:
            width: Output frame width
            height: Output frame height
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.is_running = False
        self.stream_thread = None
        self.frame_time = 1.0 / fps  # Time per frame in seconds
        self.last_frame_time = 0
        self.current_frame = None
        self.lock = threading.Lock()
    
    def start(self) -> bool:
        """Start the virtual camera stream.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            # Initialize virtual camera
            self.cam = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            
            print(f"Virtual camera started: {self.cam.device}")
            print(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}")
            
            # Create initial black frame
            self.current_frame = np.zeros((self.height, self.width, 3), np.uint8)
            
            # Start streaming thread
            self.is_running = True
            self.stream_thread = threading.Thread(target=self._stream_loop)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting virtual camera: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the virtual camera stream."""
        self.is_running = False
        
        if self.stream_thread is not None:
            self.stream_thread.join(timeout=1.0)
            self.stream_thread = None
        
        if self.cam is not None:
            self.cam.close()
            self.cam = None
    
    def update_frame(self, frame: np.ndarray) -> None:
        """Update the current frame being streamed.
        
        Args:
            frame: New frame to stream (BGR format)
        """
        if frame is None or frame.size == 0:
            return
        
        # Resize frame if needed
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Update current frame thread-safely
        with self.lock:
            self.current_frame = frame.copy()
    
    def _stream_loop(self) -> None:
        """Main streaming loop running in a separate thread."""
        while self.is_running:
            try:
                # Calculate time to next frame
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                sleep_time = max(0, self.frame_time - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Get current frame thread-safely
                with self.lock:
                    frame_to_send = self.current_frame.copy() if self.current_frame is not None else None
                
                # Send frame to virtual camera
                if frame_to_send is not None and self.cam is not None:
                    self.cam.send(frame_to_send)
                    self.cam.sleep_until_next_frame()
                
                self.last_frame_time = time.time()
            except Exception as e:
                print(f"Error in virtual camera stream: {e}")
                time.sleep(0.1)  # Sleep on error to prevent rapid error loops

class AvatarRenderer:
    """Renderer for the avatar with lip sync."""
    
    def __init__(self, width: int = 1280, height: int = 720):
        """Initialize the avatar renderer.
        
        Args:
            width: Output frame width
            height: Output frame height
        """
        self.width = width
        self.height = height
        self.background_color = (0, 0, 0)  # Black background
        self.avatar_image = None
        self.avatar_landmarks = None
        self.mouth_indices = []  # Indices of mouth landmarks
        
        # Initialize rendering parameters
        self.line_thickness = 2
        self.point_size = 3
        self.mouth_color = (0, 0, 255)  # Red for mouth
        self.face_color = (0, 255, 0)  # Green for face
    
    def set_avatar_image(self, image: np.ndarray) -> None:
        """Set the base avatar image.
        
        Args:
            image: Base avatar image (BGR format)
        """
        if image is not None and image.size > 0:
            # Resize if needed
            if image.shape[0] != self.height or image.shape[1] != self.width:
                self.avatar_image = cv2.resize(image, (self.width, self.height))
            else:
                self.avatar_image = image.copy()
    
    def set_mouth_indices(self, indices: list) -> None:
        """Set the indices of mouth landmarks.
        
        Args:
            indices: List of landmark indices corresponding to mouth region
        """
        self.mouth_indices = indices
    
    def render_frame(self, landmarks: np.ndarray) -> np.ndarray:
        """Render a frame with the avatar and updated lip position.
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            numpy.ndarray: Rendered frame
        """
        # Create base frame
        if self.avatar_image is not None:
            frame = self.avatar_image.copy()
        else:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = self.background_color
        
        # Draw landmarks if available
        if landmarks is not None and landmarks.size > 0:
            # Update avatar landmarks
            self.avatar_landmarks = landmarks
            
            # Draw mouth landmarks with special emphasis
            if self.mouth_indices and len(landmarks) > max(self.mouth_indices):
                mouth_points = landmarks[self.mouth_indices].astype(np.int32)
                
                # Draw mouth contour
                for i in range(len(mouth_points) - 1):
                    cv2.line(frame, 
                             (mouth_points[i][0], mouth_points[i][1]),
                             (mouth_points[i+1][0], mouth_points[i+1][1]),
                             self.mouth_color, 
                             self.line_thickness)
                
                # Connect last point to first to close the contour
                cv2.line(frame,
                         (mouth_points[-1][0], mouth_points[-1][1]),
                         (mouth_points[0][0], mouth_points[0][1]),
                         self.mouth_color,
                         self.line_thickness)
                
                # Draw points
                for point in mouth_points:
                    cv2.circle(frame, (point[0], point[1]), self.point_size, self.mouth_color, -1)
        
        return frame

class AvatarVirtualCamera:
    """Combined avatar renderer and virtual camera."""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """Initialize the avatar virtual camera.
        
        Args:
            width: Output frame width
            height: Output frame height
            fps: Frames per second
        """
        self.renderer = AvatarRenderer(width, height)
        self.virtual_camera = VirtualCamera(width, height, fps)
        self.is_running = False
        self.update_thread = None
        self.lock = threading.Lock()
        self.latest_landmarks = None
    
    def start(self) -> bool:
        """Start the avatar virtual camera.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.virtual_camera.start():
            return False
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the avatar virtual camera."""
        self.is_running = False
        
        if self.update_thread is not None:
            self.update_thread.join(timeout=1.0)
            self.update_thread = None
        
        self.virtual_camera.stop()
    
    def set_avatar_image(self, image: np.ndarray) -> None:
        """Set the base avatar image.
        
        Args:
            image: Base avatar image (BGR format)
        """
        self.renderer.set_avatar_image(image)
    
    def set_mouth_indices(self, indices: list) -> None:
        """Set the indices of mouth landmarks.
        
        Args:
            indices: List of landmark indices corresponding to mouth region
        """
        self.renderer.set_mouth_indices(indices)
    
    def update_landmarks(self, landmarks: np.ndarray) -> None:
        """Update the facial landmarks for lip sync.
        
        Args:
            landmarks: Facial landmarks array
        """
        with self.lock:
            self.latest_landmarks = landmarks.copy() if landmarks is not None else None
    
    def _update_loop(self) -> None:
        """Main update loop running in a separate thread."""
        while self.is_running:
            try:
                # Get latest landmarks thread-safely
                with self.lock:
                    landmarks = self.latest_landmarks.copy() if self.latest_landmarks is not None else None
                
                # Render frame with current landmarks
                frame = self.renderer.render_frame(landmarks)
                
                # Update virtual camera frame
                self.virtual_camera.update_frame(frame)
                
                # Sleep a bit to prevent CPU hogging
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in avatar update loop: {e}")
                time.sleep(0.1)  # Sleep on error to prevent rapid error loops