"""Main application window for Peeka.

This module implements the main UI window using PyQt5, providing controls
for video preview and avatar configuration.
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QGroupBox, QSpinBox, QTabWidget,
    QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon
from src.capture.video import VideoCapture
from src.capture.audio import AudioCapture
import cv2

class PeekaApp(QMainWindow):
    """Main application window for the Peeka avatar system."""

    def __init__(self, video_capture: VideoCapture, audio_capture: AudioCapture):
        """Initialize the main application window.

        Args:
            video_capture: VideoCapture instance for camera input
            audio_capture: AudioCapture instance for audio input
        """
        super().__init__()
        self.video_capture = video_capture
        self.audio_capture = audio_capture

        self.setWindowTitle('Peeka Avatar System')
        self.setGeometry(100, 100, 1024, 768)  # Increased window size for better layout

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        main_layout.setSpacing(10)  # Space between panels

        # Left panel for video preview
        left_panel = QWidget()
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expansion
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins
        left_layout.setSpacing(5)  # Tighter spacing
        
        # Create video preview label with proportional size
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)  # Larger minimum size
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label)
        
        # Status bar - more compact
        status_box = QGroupBox("Status")
        status_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_layout = QHBoxLayout(status_box)
        status_layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins
        self.status_label = QLabel('Status: Initializing...')
        self.fps_label = QLabel('FPS: --')
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.fps_label)
        left_layout.addWidget(status_box)
        
        main_layout.addWidget(left_panel, stretch=7)  # Give more space to video

        # Right panel with tabs for controls
        right_panel = QTabWidget()
        right_panel.setFixedWidth(250)  # Fixed width instead of minimum
        right_panel.setMaximumHeight(300)  # Increased height for additional controls
        right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Training Tab
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        training_layout.setContentsMargins(2, 2, 2, 2)
        training_layout.setSpacing(5)

        # Training controls
        training_group = QGroupBox("Avatar Training")
        training_box_layout = QVBoxLayout(training_group)
        training_box_layout.setContentsMargins(5, 5, 5, 5)

        # Training button
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.open_training_window)
        training_box_layout.addWidget(self.train_button)

        # Training status
        self.training_status = QLabel("Status: Not trained")
        training_box_layout.addWidget(self.training_status)

        training_layout.addWidget(training_group)
        right_panel.addTab(training_tab, "Training")

        # Camera Settings Tab - more compact
        camera_tab = QWidget()
        camera_layout = QVBoxLayout(camera_tab)
        camera_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        camera_layout.setSpacing(1)  # Minimal spacing

        # Camera Settings Group - ultra compact
        camera_group = QGroupBox("Camera Settings")
        camera_box_layout = QVBoxLayout(camera_group)
        camera_box_layout.setContentsMargins(2, 2, 2, 2)  # Minimal margins
        camera_box_layout.setSpacing(1)  # Minimal spacing

        # Camera Selection - horizontal layout
        camera_row = QHBoxLayout()
        camera_label = QLabel("Camera:")
        camera_label.setFixedWidth(60)  # Fixed width for alignment
        self.camera_combo = QComboBox()
        self.camera_combo.setFixedHeight(22)  # Smaller height
        self.available_cameras = self.video_capture.list_devices()
        camera_names = [device['name'] for device in self.available_cameras]
        self.camera_combo.addItems(camera_names)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        camera_row.addWidget(camera_label)
        camera_row.addWidget(self.camera_combo)
        camera_box_layout.addLayout(camera_row)

        # Resolution Selection - horizontal layout
        resolution_row = QHBoxLayout()
        resolution_label = QLabel("Resolution:")
        resolution_label.setFixedWidth(60)  # Fixed width for alignment
        self.resolution_combo = QComboBox()
        self.resolution_combo.setFixedHeight(22)  # Smaller height
        self.resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        self.resolution_combo.addItems([f"{w}x{h}" for w, h in self.resolutions])
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_changed)
        resolution_row.addWidget(resolution_label)
        resolution_row.addWidget(self.resolution_combo)
        camera_box_layout.addLayout(resolution_row)

        camera_layout.addWidget(camera_group)
        right_panel.addTab(camera_tab, "Camera")
        
        main_layout.addWidget(right_panel)

        # Setup video update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        self.timer.start(33)  # ~30 FPS

        # Initialize camera
        if not self.video_capture.start():
            QMessageBox.warning(self, "Camera Error", 
                "Failed to initialize camera. Please check your camera connection.")
            self.status_label.setText('Status: No camera')

    def on_camera_changed(self, index: int) -> None:
        """Handle camera device selection change."""
        if 0 <= index < len(self.available_cameras):
            try:
                device_id = int(self.available_cameras[index]['id'])
                self.video_capture.release()
                self.video_capture.device_id = device_id
                if not self.video_capture.start():
                    raise Exception("Failed to start camera")
                self.status_label.setText('Status: Running')
            except Exception as e:
                QMessageBox.warning(self, "Camera Error", 
                    f"Failed to switch camera: {str(e)}")
                self.status_label.setText('Status: Camera error')

    def on_resolution_changed(self, index: int) -> None:
        """Handle resolution selection change."""
        if 0 <= index < len(self.resolutions):
            try:
                width, height = self.resolutions[index]
                self.video_capture.set_resolution(width, height)
            except Exception as e:
                QMessageBox.warning(self, "Resolution Error", 
                    f"Failed to change resolution: {str(e)}")

    def update_video_frame(self):
        """Update the video preview with the latest frame."""
        try:
            success, frame = self.video_capture.read_frame()
            if success:
                # Convert frame to RGB format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape

                # Convert to QImage and display
                qt_image = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled_pixmap)

                # Update FPS display
                self.fps_label.setText(f'FPS: {self.video_capture.get_fps():.1f}')
                # Update resolution display
                current_width, current_height = self.video_capture.current_resolution
                self.status_label.setText(f'Resolution: {current_width}x{current_height}')
            else:
                self.status_label.setText('Status: No frame')
                self.fps_label.setText('FPS: --')
        except Exception as e:
            self.status_label.setText('Status: Error')
            print(f"Frame update error: {e}")

    def open_training_window(self):
        """Open the training window for facial feature capture."""
        try:
            from src.ui.training_window import TrainingWindow
            self.training_window = TrainingWindow()
            self.training_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Training Error", 
                f"Failed to open training window: {str(e)}")

    def closeEvent(self, event):
        """Handle application close event."""
        self.timer.stop()
        super().closeEvent(event)