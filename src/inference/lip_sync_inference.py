\"""Lip Sync Inference module for real-time audio-driven facial animation.

This module handles real-time inference using the trained lip sync model,
processing audio input and generating corresponding facial landmark positions.
"""

import os
import torch
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any

from src.models.lip_sync_model import LipSyncModel, LipSyncTrainer

class LipSyncInference:
    """Real-time inference class for lip synchronization."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model file (optional)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = LipSyncTrainer()
        self.model = None
        self.is_running = False
        self.inference_thread = None
        self.audio_buffer = queue.Queue(maxsize=100)
        self.result_buffer = queue.Queue(maxsize=10)
        self.smoothing_window = 3  # Frames for temporal smoothing
        self.previous_predictions = []
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained lip sync model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.trainer.load_model(model_path)
            self.model = self.trainer.model
            print(f"Loaded lip sync model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading lip sync model: {e}")
            return False
    
    def start(self) -> bool:
        """Start the inference engine.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.model is None:
            print("Error: No model loaded. Please load a model first.")
            return False
        
        if self.is_running:
            return True  # Already running
        
        self.is_running = True
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        return True
    
    def stop(self) -> None:
        """Stop the inference engine."""
        self.is_running = False
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=1.0)
            self.inference_thread = None
    
    def add_audio_features(self, audio_features: np.ndarray) -> None:
        """Add audio features to the processing queue.
        
        Args:
            audio_features: MFCC features extracted from audio
        """
        if not self.audio_buffer.full():
            self.audio_buffer.put({
                'features': audio_features,
                'timestamp': time.time()
            })
    
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the latest landmark prediction.
        
        Returns:
            Dict containing predicted landmarks or None if no prediction available
        """
        if self.result_buffer.empty():
            return None
        
        # Get all available predictions and return only the most recent one
        latest_prediction = None
        while not self.result_buffer.empty():
            latest_prediction = self.result_buffer.get()
        
        return latest_prediction
    
    def _apply_temporal_smoothing(self, current_prediction: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to predictions for more natural movement.
        
        Args:
            current_prediction: Current frame's landmark predictions
            
        Returns:
            numpy.ndarray: Smoothed landmark predictions
        """
        # Add current prediction to history
        self.previous_predictions.append(current_prediction)
        
        # Keep only the last N predictions
        if len(self.previous_predictions) > self.smoothing_window:
            self.previous_predictions.pop(0)
        
        # Apply weighted average (more weight to recent frames)
        weights = np.linspace(0.5, 1.0, len(self.previous_predictions))
        weights = weights / weights.sum()  # Normalize
        
        smoothed = np.zeros_like(current_prediction)
        for i, pred in enumerate(self.previous_predictions):
            smoothed += weights[i] * pred
        
        return smoothed
    
    def _inference_loop(self) -> None:
        """Main inference loop running in a separate thread."""
        while self.is_running:
            try:
                if not self.audio_buffer.empty():
                    # Get audio features from buffer
                    audio_data = self.audio_buffer.get()
                    audio_features = audio_data['features']
                    
                    # Prepare input tensor
                    # Add batch and sequence dimensions if needed
                    if len(audio_features.shape) == 2:
                        # Already has sequence dimension, just add batch
                        input_tensor = torch.FloatTensor(audio_features).unsqueeze(0)
                    else:
                        # Add both batch and sequence dimensions
                        input_tensor = torch.FloatTensor(audio_features).unsqueeze(0).unsqueeze(0)
                    
                    # Generate prediction
                    with torch.no_grad():
                        input_tensor = input_tensor.to(self.device)
                        prediction = self.model(input_tensor)
                        prediction_np = prediction.cpu().numpy().squeeze()
                    
                    # Apply temporal smoothing
                    smoothed_prediction = self._apply_temporal_smoothing(prediction_np)
                    
                    # Add to result buffer if not full
                    if not self.result_buffer.full():
                        self.result_buffer.put({
                            'landmarks': smoothed_prediction,
                            'timestamp': time.time(),
                            'original_timestamp': audio_data['timestamp']
                        })
                else:
                    # Sleep a bit to prevent CPU hogging
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error in lip sync inference: {e}")
                time.sleep(0.1)  # Sleep on error to prevent rapid error loops
    
    def process_single_frame(self, audio_features: np.ndarray) -> np.ndarray:
        """Process a single frame for non-real-time applications.
        
        Args:
            audio_features: MFCC features for a single frame
            
        Returns:
            numpy.ndarray: Predicted facial landmarks
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Prepare input tensor
        input_tensor = torch.FloatTensor(audio_features).unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            prediction = self.model(input_tensor)
            prediction_np = prediction.cpu().numpy().squeeze()
        
        return prediction_np