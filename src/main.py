"""Main entry point for the Peeka application.

This module initializes and coordinates the core components of the application,
including video capture, audio processing, and virtual camera output.
"""

import sys
from typing import Optional

from capture.video import VideoCapture
from capture.audio import AudioCapture
from virtual_camera import VirtualCamera
from ui.app import PeekaApp
from PyQt5.QtWidgets import QApplication

def initialize_components() -> tuple[Optional[VideoCapture], Optional[AudioCapture]]:
    """Initialize video and audio capture components.

    Returns:
        tuple: Initialized video and audio capture instances
    """
    video_capture = VideoCapture()
    audio_capture = AudioCapture()

    if not video_capture.start():
        print("Error: Failed to initialize video capture")
        return None, None

    if not audio_capture.start():
        print("Error: Failed to initialize audio capture")
        video_capture.release()
        return None, None

    return video_capture, audio_capture

def main() -> int:
    """Main application entry point.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Initialize Qt application
    app = QApplication(sys.argv)

    # Initialize capture components
    video_capture, audio_capture = initialize_components()
    if not (video_capture and audio_capture):
        return 1

    try:
        # Create and show the main application window
        main_window = PeekaApp(video_capture, audio_capture)
        main_window.show()

        # Start the application event loop
        return app.exec_()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        # Ensure resources are properly released
        if video_capture:
            video_capture.release()
        if audio_capture:
            audio_capture.release()

if __name__ == '__main__':
    sys.exit(main())