"""Audio capture module for real-time audio input processing.

This module provides functionality for capturing and preprocessing audio input
from system microphones, with support for noise reduction and normalization.
"""

import numpy as np
import pyaudio
from typing import Optional, Tuple, Dict
import librosa

class AudioCapture:
    """Handles audio capture and preprocessing from system microphones."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024,
                 channels: int = 1, format_type: int = pyaudio.paFloat32):
        """Initialize the audio capture system.

        Args:
            sample_rate: Audio sampling rate in Hz (default: 16000)
            chunk_size: Number of frames per buffer (default: 1024)
            channels: Number of audio channels (default: 1 for mono)
            format_type: PyAudio format type (default: 32-bit float)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format_type = format_type
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start(self, device_index: Optional[int] = None) -> bool:
        """Start audio capture from the specified device.

        Args:
            device_index: Optional device index (default: None for default device)

        Returns:
            bool: True if capture started successfully, False otherwise
        """
        try:
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            return True
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            return False

    def read_frame(self) -> Optional[bytes]:
        """Read a frame of audio data from the stream.

        Returns:
            bytes or None: The captured audio data if successful, None otherwise
        """
        if not self.stream:
            return None

        try:
            return self.stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            print(f"Error reading audio frame: {e}")
            return None

    def preprocess_audio(self, audio_chunk: np.ndarray,
                        normalize: bool = True) -> np.ndarray:
        """Apply preprocessing to the audio chunk.

        Args:
            audio_chunk: Input audio data
            normalize: Whether to normalize the audio (default: True)

        Returns:
            np.ndarray: Preprocessed audio data
        """
        if normalize:
            audio_chunk = librosa.util.normalize(audio_chunk)
        return audio_chunk

    def get_audio_devices(self) -> Dict[int, str]:
        """Get a list of available audio input devices.

        Returns:
            Dict[int, str]: Dictionary mapping device indices to their names
        """
        devices = {}
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices[i] = device_info['name']
        return devices

    def stop(self) -> None:
        """Stop the audio capture stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def release(self) -> None:
        """Release the audio capture resources."""
        self.stop()
        if self.audio:
            self.audio.terminate()