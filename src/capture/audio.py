"""Audio capture module for real-time audio input processing.

This module provides functionality for capturing and preprocessing audio input
from system microphones, with support for feature extraction needed for lip sync.
"""

import pyaudio
import numpy as np
import wave
import threading
import time
import librosa
import queue
from typing import Optional, Tuple, List, Dict, Any

class AudioCapture:
    """Audio capture class for handling microphone input."""

    def __init__(self):
        """Initialize audio capture with default settings."""
        self.device_id = None  # Default device
        self.stream = None
        self.audio = None
        self.is_recording = False
        self.frames = []
        self.lock = threading.Lock()
        self.sample_rate = 16000  # 16kHz is good for speech processing
        self.channels = 1  # Mono for speech processing
        self.format = pyaudio.paInt16
        self.chunk_size = 1024
        self.audio_buffer = queue.Queue(maxsize=100)  # Buffer for real-time processing
        self.feature_buffer = queue.Queue(maxsize=100)  # Buffer for extracted features
        self.feature_thread = None
        self.feature_processing = False

    def list_devices(self) -> list:
        """List available audio input devices.

        Returns:
            list: List of dictionaries containing device info
        """
        available_devices = []
        try:
            self.audio = pyaudio.PyAudio()
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(num_devices):
                device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    available_devices.append({
                        'id': i,
                        'name': device_info.get('name'),
                        'channels': device_info.get('maxInputChannels'),
                        'sample_rate': int(device_info.get('defaultSampleRate'))
                    })
        except Exception as e:
            print(f"Error listing audio devices: {e}")
        
        return available_devices if available_devices else [{'id': None, 'name': 'Default Microphone'}]

    def start(self) -> bool:
        """Start audio capture.

        Returns:
            bool: True if capture started successfully, False otherwise
        """
        try:
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.frames = []
            
            # Start feature extraction thread
            self.feature_processing = True
            self.feature_thread = threading.Thread(target=self._process_audio_features)
            self.feature_thread.daemon = True
            self.feature_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status) -> Tuple[bytes, int]:
        """Callback function for audio stream.

        Args:
            in_data: Input audio data
            frame_count: Number of frames
            time_info: Time information
            status: Status flag

        Returns:
            Tuple containing the data and pyaudio flag
        """
        try:
            # Add the audio data to our buffer for processing
            if not self.audio_buffer.full():
                self.audio_buffer.put(in_data)
            
            # Also store frames for recording if needed
            with self.lock:
                if self.is_recording:
                    self.frames.append(in_data)
        except Exception as e:
            print(f"Error in audio callback: {e}")
        
        return (in_data, pyaudio.paContinue)

    def _process_audio_features(self) -> None:
        """Process audio data to extract features for lip sync."""
        while self.feature_processing:
            try:
                if not self.audio_buffer.empty():
                    # Get audio data from buffer
                    audio_data = self.audio_buffer.get()
                    
                    # Convert to numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Extract features (MFCC for phoneme detection)
                    if len(audio_np) > 0:
                        mfcc = librosa.feature.mfcc(
                            y=audio_np, 
                            sr=self.sample_rate, 
                            n_mfcc=13,  # Standard for speech recognition
                            hop_length=int(self.sample_rate * 0.01)  # 10ms hop
                        )
                        
                        # Add to feature buffer if not full
                        if not self.feature_buffer.full():
                            self.feature_buffer.put({
                                'mfcc': mfcc,
                                'timestamp': time.time()
                            })
                else:
                    # Sleep a bit to prevent CPU hogging
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error processing audio features: {e}")
                time.sleep(0.1)  # Sleep on error to prevent rapid error loops

    def get_latest_features(self) -> Optional[Dict[str, Any]]:
        """Get the latest extracted audio features.

        Returns:
            Dict containing audio features or None if no features available
        """
        if self.feature_buffer.empty():
            return None
        
        # Get all available features and return only the most recent one
        latest_feature = None
        while not self.feature_buffer.empty():
            latest_feature = self.feature_buffer.get()
        
        return latest_feature

    def save_recording(self, filename: str) -> bool:
        """Save the recorded audio to a WAV file.

        Args:
            filename: Path to save the WAV file

        Returns:
            bool: True if saved successfully, False otherwise
        """
        if not self.frames:
            return False
        
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                with self.lock:
                    wf.writeframes(b''.join(self.frames))
            return True
        except Exception as e:
            print(f"Error saving audio recording: {e}")
            return False

    def release(self) -> None:
        """Release audio resources."""
        self.is_recording = False
        self.feature_processing = False
        
        if self.feature_thread is not None:
            self.feature_thread.join(timeout=1.0)
        
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None