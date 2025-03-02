"""Training window module for facial feature capture.

This module provides a dedicated UI for the training phase, showing real-time
facial landmark visualization and guiding users through the capture process.
"""

import os
import time
import wave
import pyaudio
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QGroupBox, QMessageBox, QScrollArea,
    QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import mediapipe as mp

from src.capture.training_capture import TrainingDataCapture
from src.capture.setup_dlib import download_shape_predictor
from src.capture.audio import AudioCapture

class TrainingWindow(QMainWindow):
    """Training window for facial feature capture session."""

    def __init__(self):
        """Initialize the training window."""
        super().__init__()
        # Initialize timer before any potential exceptions
        self.timer = QTimer()
        self.countdown_timer = QTimer()
        try:
            # Initialize audio system first
            audio = pyaudio.PyAudio()
            audio.terminate()
            
            # Download and setup dlib shape predictor
            download_shape_predictor()
            self.training_capture = TrainingDataCapture()
            # Initialize MediaPipe face mesh with consistent configuration
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,  # Match TrainingDataCapture configuration
                min_tracking_confidence=0.5,  # Match TrainingDataCapture configuration
                refine_landmarks=False  # Match TrainingDataCapture configuration for better performance
            )
            self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            self.setup_ui()
        except Exception as e:
            error_msg = str(e)
            if "PortAudio" in error_msg or "Audio" in error_msg:
                error_msg = "Failed to initialize audio system. Please check your audio devices and permissions."
            QMessageBox.critical(self, "Initialization Error", error_msg)
            self.close()

    def setup_ui(self):
        """Setup the training window UI components."""
        self.setWindowTitle('Facial Feature Training')
        self.setGeometry(100, 100, 1024, 768)  # Increased window size

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)  # Add spacing between components
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add padding

        # Create a scroll area for instructions
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)  # Limit height of instructions
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Instructions group
        instructions_group = QGroupBox('Instructions')
        instructions_layout = QVBoxLayout(instructions_group)
        instructions_text = QLabel(
            'Please follow these steps for the 5-minute training session:\n'
            '1. Position your face in the center of the frame\n'
            '2. Ensure good lighting conditions\n'
            '3. Make various facial expressions during capture:\n'
            '   - Smile, frown, raise eyebrows\n'
            '   - Open and close mouth\n'
            '   - Look in different directions\n'
            '4. Stay relatively still but natural\n'
            '5. Speak normally to capture voice patterns\n\n'
            'Training Prompts (read aloud):\n'
            '• Express happiness: "I am so excited about this wonderful day!"\n'
            '• Show surprise: "Wow! I cannot believe what I just saw!"\n'
            '• Display confusion: "I am not quite sure I understand that concept."\n'
            '• Show determination: "I will definitely achieve my goals today."\n'
            '• Express thoughtfulness: "Let me think about that for a moment..."\n'
            '• Read naturally: "The quick brown fox jumps over the lazy dog."\n'
            '• Practice emotions: "Sometimes I feel happy, sometimes sad, and that is okay."\n'
            '• Show emphasis: "This is REALLY important to remember!"\n'
            '• Express agreement: "Yes, I completely agree with that point."\n'
            '• Show disagreement: "No, I do not think that is correct."'
        )
        instructions_text.setAlignment(Qt.AlignLeft)
        instructions_text.setWordWrap(True)  # Enable word wrap
        instructions_layout.addWidget(instructions_text)
        scroll_layout.addWidget(instructions_group)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # Create horizontal layout for preview and controls
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Left side: Video preview with landmark visualization
        preview_group = QGroupBox('Live Preview with Facial Landmarks')
        preview_layout = QVBoxLayout(preview_group)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.video_label)
        content_layout.addWidget(preview_group, stretch=2)  # Give more space to preview

        # Right side: Controls, progress, and metrics
        controls_container = QWidget()
        controls_container_layout = QVBoxLayout(controls_container)
        controls_container_layout.setSpacing(10)

        # Add metrics display
        metrics_group = QGroupBox('Facial Metrics')
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_labels = {
            'left_eye': QLabel('Left Eye Ratio: 0.00'),
            'right_eye': QLabel('Right Eye Ratio: 0.00'),
            'mouth': QLabel('Mouth Ratio: 0.00'),
            'left_brow': QLabel('Left Brow Height: 0.00'),
            'right_brow': QLabel('Right Brow Height: 0.00')
        }
        for label in self.metrics_labels.values():
            metrics_layout.addWidget(label)
        controls_container_layout.addWidget(metrics_group)

        # Add quality indicators
        quality_group = QGroupBox('Frame Quality')
        quality_layout = QVBoxLayout(quality_group)
        self.quality_indicators = {
            'brightness': QProgressBar(),
            'contrast': QProgressBar(),
            'sharpness': QProgressBar(),
            'stability': QProgressBar()
        }
        for name, bar in self.quality_indicators.items():
            label = QLabel(f'{name.title()}: ')
            bar.setMinimum(0)
            bar.setMaximum(100)
            bar_layout = QHBoxLayout()
            bar_layout.addWidget(label)
            bar_layout.addWidget(bar)
            quality_layout.addLayout(bar_layout)
        controls_container_layout.addWidget(quality_group)

        # Progress group
        progress_group = QGroupBox('Capture Progress')
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        self.status_label = QLabel('Ready to start capture')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)  # Enable word wrap for status
        progress_layout.addWidget(self.status_label)
        controls_container_layout.addWidget(progress_group)

        # Duration selection and control buttons
        controls_group = QGroupBox('Capture Controls')
        controls_layout = QVBoxLayout(controls_group)

        # Duration selection
        duration_layout = QHBoxLayout()
        duration_label = QLabel('Duration (minutes):')
        duration_layout.addWidget(duration_label)

        self.duration_buttons = []
        for mins in range(1, 6):
            btn = QPushButton(str(mins))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mins: self.select_duration(m))
            duration_layout.addWidget(btn)
            self.duration_buttons.append(btn)

        # Default to 5 minutes
        self.duration_buttons[-1].setChecked(True)
        self.selected_duration = 5
        controls_layout.addLayout(duration_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('Start Capture')
        self.start_button.clicked.connect(self.start_capture)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.load_dataset_button = QPushButton('Load Existing Dataset')
        self.load_dataset_button.clicked.connect(self.load_existing_dataset)
        button_layout.addWidget(self.load_dataset_button)

        controls_layout.addLayout(button_layout)
        controls_container_layout.addWidget(controls_group)
        content_layout.addWidget(controls_container, stretch=1)  # Less space for controls

        # Setup timers with optimized intervals
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_capture)
        self.timer.setInterval(33)  # ~30 FPS for smoother visualization

        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.setInterval(1000)  # 1 second
        self.remaining_time = 0
        self.frame_skip_counter = 0

        # Initialize MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def select_duration(self, minutes: int) -> None:
        """Handle duration selection."""
        self.selected_duration = minutes
        for btn in self.duration_buttons:
            btn.setChecked(int(btn.text()) == minutes)

    def update_countdown(self) -> None:
        """Update the countdown timer display."""
        self.remaining_time -= 1
        minutes = self.remaining_time // 60
        seconds = self.remaining_time % 60
        self.status_label.setText(
            f'Recording in progress... {minutes:02d}:{seconds:02d} remaining\n'
            'Keep making different expressions')

        # Check if we have enough frames before stopping
        expected_min = int(self.training_capture.target_duration * 20 * 0.4)  # 40% of expected frames
        if self.remaining_time <= 0:
            if self.training_capture.frame_count >= expected_min:
                self.stop_capture()
            else:
                # Extend recording time by 30 seconds if we don't have enough frames
                self.remaining_time = 30
                self.status_label.setText(
                    f'Extended recording time to collect more frames...\n'
                    f'Current: {self.training_capture.frame_count}, Need: {expected_min}')

    def start_capture(self):
        """Start the training capture session."""
        try:
            self.training_capture.target_duration = self.selected_duration * 60
            success, message = self.training_capture.start_recording()
            if success:
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                for btn in self.duration_buttons:
                    btn.setEnabled(False)
                
                self.remaining_time = self.selected_duration * 60
                self.status_label.setText('Recording in progress... Keep making different expressions')
                self.timer.start()  # Start the capture timer
                self.countdown_timer.start()
            else:
                QMessageBox.warning(self, "Capture Error", f"Failed to start capture: {message}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def stop_capture(self):
        """Stop the training capture session."""
        try:
            # Allow a grace period for final frames to be processed
            if self.training_capture.recording:
                self.status_label.setText('Finalizing capture...')
                QTimer.singleShot(500, self._finalize_capture)  # 500ms grace period
            else:
                self._finalize_capture()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def _finalize_capture(self):
        """Finalize capture with improved feedback and show training panel."""
        try:
            session_dir = self.training_capture.stop_recording()
            self.timer.stop()
            self.countdown_timer.stop()
            
            # Enable UI controls
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            for btn in self.duration_buttons:
                btn.setEnabled(True)
            
            face_frames_dir = os.path.join(session_dir, 'face_frames')
            frame_count = len([f for f in os.listdir(face_frames_dir) if f.endswith('.jpg')])
            
            if frame_count < 60:  # Minimum 2 seconds of footage
                QMessageBox.warning(self, "Insufficient Data", 
                    f'Only captured {frame_count} frames. Please record at least 2 seconds.')
                return

            if self.training_capture.validate_recording(session_dir):
                result = QMessageBox.information(self, "Success", 
                    f'Training data capture completed successfully!\nCaptured {frame_count} frames.\n\nWould you like to proceed to model training?',
                    QMessageBox.Yes | QMessageBox.No)
                
                if result == QMessageBox.Yes:
                    self.show_training_results_panel(session_dir)
            else:
                QMessageBox.warning(self, "Validation Failed",
                    'Training data did not meet quality requirements.\n'
                    'Please ensure:\n'
                    '- Your face is clearly visible\n'
                    '- Lighting is adequate\n'
                    '- You make various expressions\n'
                    '- Camera is stable')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            
    def load_existing_dataset(self):
        """Load an existing dataset for training without capturing new video."""
        try:
            # Get the base training data directory
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
            
            # Show directory selection dialog
            session_dir = QFileDialog.getExistingDirectory(
                self, "Select Training Session Directory", base_dir,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
            )
            
            if not session_dir:
                return  # User canceled
            
            # Validate that this is a proper training session directory
            landmarks_dir = os.path.join(session_dir, 'landmarks_data')
            if not os.path.exists(landmarks_dir):
                QMessageBox.warning(self, "Invalid Directory", 
                    "The selected directory does not contain landmark data.\n"
                    "Please select a valid training session directory.")
                return
            
            # Check if there are enough landmark files
            landmark_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.json')]
            if len(landmark_files) < 60:  # Minimum requirement
                QMessageBox.warning(self, "Insufficient Data", 
                    f"The selected directory contains only {len(landmark_files)} landmark files.\n"
                    "At least 60 files are required for training.")
                return
            
            # Proceed to training results panel with the selected directory
            self.show_training_results_panel(session_dir)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def show_training_results_panel(self, session_dir):
        """Show the training results panel with the captured data.
        
        Args:
            session_dir: Directory containing the captured training data
        """
        try:
            from src.ui.training_results_panel import TrainingResultsPanel
            
            # Create training results panel
            self.training_results_panel = TrainingResultsPanel(session_dir=session_dir)
            
            # Create a new window to display the panel
            self.training_results_window = QMainWindow()
            self.training_results_window.setWindowTitle("Training Results and Model Training")
            self.training_results_window.setCentralWidget(self.training_results_panel)
            self.training_results_window.setGeometry(100, 100, 1024, 768)
            self.training_results_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open training results panel: {str(e)}")

    def update_capture(self):
        """Update the video preview and progress."""
        try:
            # Implement frame skipping
            self.frame_skip_counter += 1
            if self.frame_skip_counter % 2 != 0:  # Process every other frame
                return

            success, frame = self.training_capture.video_capture.read_frame()
            if success:
                try:
                    # Convert to RGB for MediaPipe processing
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Check lighting conditions
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = cv2.mean(gray_frame)[0]
                    if avg_brightness < 40:  # Too dark
                        self.status_label.setText('Warning: Poor lighting conditions. Please increase lighting.')
                    elif avg_brightness > 220:  # Too bright
                        self.status_label.setText('Warning: Too bright. Please reduce lighting.')
                    
                    # Process frame with MediaPipe
                    results = self.face_mesh.process(rgb_frame)
                    
                    # Create a copy for visualization
                    display_frame = frame.copy()
                    
                    if results.multi_face_landmarks:
                        # Draw the face mesh on the frame
                        for face_landmarks in results.multi_face_landmarks:
                            # Draw the face mesh connections
                            self.mp_drawing.draw_landmarks(
                                image=display_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=self.drawing_spec,
                                connection_drawing_spec=self.drawing_spec
                            )
                            
                            # Add labels for key facial features
                            feature_labels = {
                                'Right Eye': 33,    # Right eye outer corner
                                'Left Eye': 263,    # Left eye outer corner
                                'Nose Tip': 1,      # Nose tip
                                'Mouth': 61,        # Mouth right corner
                                'Jaw': 152          # Chin
                            }
                            
                            # Draw labels at landmark positions
                            img_h, img_w, _ = display_frame.shape
                            for label, idx in feature_labels.items():
                                landmark = face_landmarks.landmark[idx]
                                pos = (int(landmark.x * img_w), int(landmark.y * img_h))
                                cv2.putText(display_frame, label, (pos[0], pos[1]-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                    # Convert frame to Qt format and display
                    rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_display.shape
                    qt_image = QImage(rgb_display.data, w, h, w * ch, QImage.Format_RGB888)
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                        self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.video_label.setPixmap(scaled_pixmap)

                    # Update progress, metrics, and save frame if recording
                    if self.training_capture.recording:
                        progress = min(100, (self.training_capture.frame_count / (self.training_capture.target_duration * 10)) * 100)
                        self.progress_bar.setValue(int(progress))

                        # Update facial metrics display
                        if results.multi_face_landmarks:
                            landmarks = self.training_capture.detect_facial_landmarks(frame)
                            if landmarks and 'face_metrics' in landmarks:
                                metrics = landmarks['face_metrics']
                                self.metrics_labels['left_eye'].setText(f'Left Eye Ratio: {metrics["left_eye_ratio"]:.2f}')
                                self.metrics_labels['right_eye'].setText(f'Right Eye Ratio: {metrics["right_eye_ratio"]:.2f}')
                                self.metrics_labels['mouth'].setText(f'Mouth Ratio: {metrics["mouth_aspect_ratio"]:.2f}')
                                self.metrics_labels['left_brow'].setText(f'Left Brow Height: {metrics["left_brow_height"]:.2f}')
                                self.metrics_labels['right_brow'].setText(f'Right Brow Height: {metrics["right_brow_height"]:.2f}')

                            # Update quality indicators
                            quality = self.training_capture._assess_frame_quality(frame)
                            self.quality_indicators['brightness'].setValue(int(min(quality['brightness'] / 2.55, 100)))
                            self.quality_indicators['contrast'].setValue(int(min(quality['contrast'] / 2, 100)))
                            self.quality_indicators['sharpness'].setValue(int(min(quality['sharpness'] / 10, 100)))
                            
                            # Calculate stability based on landmark movement
                            if hasattr(self, 'prev_landmarks'):
                                movement = np.mean(np.linalg.norm(
                                    np.array(landmarks['landmarks_2d']) - self.prev_landmarks, axis=1))
                                stability = max(0, min(100, 100 - movement))
                                self.quality_indicators['stability'].setValue(int(stability))
                            self.prev_landmarks = np.array(landmarks['landmarks_2d'])

                        self.training_capture.update_capture(frame)

                    # Force cleanup of Qt image objects and other resources
                    qt_image = None
                    scaled_pixmap = None
                    rgb_display = None
                    display_frame = None
                    rgb_frame = None
                    gray_frame = None
                    results = None
                finally:
                    # Ensure resources are released even if an exception occurs
                    frame = None
        except Exception as e:
            self.timer.stop()
            self.countdown_timer.stop()
            QMessageBox.critical(self, "Error", f"Capture error: {str(e)}")
            self.close()

    def closeEvent(self, event):
        """Handle window close event."""
        self.timer.stop()
        self.countdown_timer.stop()
        if hasattr(self, 'training_capture'):
            # Stop recording if in progress
            if self.training_capture.recording:
                self.training_capture.stop_recording()
            # Release video capture resources
            self.training_capture.video_capture.release()
            # Release audio capture resources
            self.training_capture.audio_capture.release()
        
        # Release face mesh resources
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        
        super().closeEvent(event)