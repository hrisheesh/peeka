"""Video and audio capture module for Peeka.

This module handles real-time video and audio capture from system devices,
providing preprocessed data for the avatar generation pipeline.
"""

from typing import Any

__version__ = "0.1.0"
from .audio import AudioCapture

__all__ = ['AudioCapture']