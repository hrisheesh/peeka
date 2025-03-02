"""Training Manager for Peeka facial feature models.

This module provides a training manager that integrates with the UI to
visualize and control the training process for facial feature models.
"""

import os
import threading
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

from .facial_feature_model import FacialFeatureDataset, FacialFeatureTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingWorker(QThread):
    """Worker thread for training models without blocking the UI."""
    
    # Define signals for progress updates
    progress_updated = pyqtSignal(int, str)
    epoch_completed = pyqtSignal(int, float, float)
    training_completed = pyqtSignal(dict)
    training_error = pyqtSignal(str)
    
    def __init__(self, trainer: FacialFeatureTrainer, dataset: FacialFeatureDataset, 
                 epochs: int = 50, batch_size: int = 32, patience: int = 10):
        """Initialize the training worker.
        
        Args:
            trainer: The model trainer instance
            dataset: The dataset to train on
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
        """
        super().__init__()
        self.trainer = trainer
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.is_running = False
        
    def run(self):
        """Run the training process in a separate thread."""
        self.is_running = True
        try:
            # Override trainer's logging to emit signals
            original_train = self.trainer.train
            
            def train_with_progress(*args, **kwargs):
                # Split dataset into train and validation
                val_split = kwargs.get('val_split', 0.2)
                val_size = int(len(self.dataset) * val_split)
                train_size = len(self.dataset) - val_size
                
                self.progress_updated.emit(0, f"Preparing dataset: {len(self.dataset)} samples")
                
                # Initialize model if not already created
                if self.trainer.model is None:
                    self.progress_updated.emit(5, "Initializing model...")
                    sample = self.dataset[0]
                    input_size = sample['input'].shape[0]
                    output_size = sample['target'].shape[0]
                    self.trainer.create_model(input_size, hidden_size=256, output_size=output_size)
                
                # Training loop with progress updates
                self.progress_updated.emit(10, f"Starting training with {train_size} samples")
                
                # Create data loaders
                train_dataset, val_dataset = torch.utils.data.random_split(
                    self.dataset, 
                    [train_size, val_size]
                )
                
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True
                )
                
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, 
                    batch_size=self.batch_size
                )
                
                best_val_loss = float('inf')
                no_improve_epochs = 0
                
                for epoch in range(self.epochs):
                    if not self.is_running:
                        break
                    
                    # Calculate progress percentage
                    progress = 10 + int(90 * (epoch / self.epochs))
                    self.progress_updated.emit(progress, f"Training epoch {epoch+1}/{self.epochs}")
                    
                    # Training phase
                    self.trainer.model.train()
                    train_loss = 0.0
                    
                    for batch in train_loader:
                        inputs = batch['input'].to(self.trainer.device)
                        targets = batch['target'].to(self.trainer.device)
                        qualities = batch['quality'].to(self.trainer.device)
                        
                        # Forward pass
                        self.trainer.optimizer.zero_grad()
                        outputs = self.trainer.model(inputs)
                        
                        # Weight loss by frame quality
                        loss = self.trainer.criterion(outputs, targets)
                        weighted_loss = (loss * qualities.view(-1, 1)).mean()
                        
                        # Backward pass and optimize
                        weighted_loss.backward()
                        self.trainer.optimizer.step()
                        
                        train_loss += loss.item() * inputs.size(0)
                    
                    train_loss /= len(train_loader.dataset)
                    
                    # Validation phase
                    self.trainer.model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch in val_loader:
                            inputs = batch['input'].to(self.trainer.device)
                            targets = batch['target'].to(self.trainer.device)
                            
                            outputs = self.trainer.model(inputs)
                            loss = self.trainer.criterion(outputs, targets)
                            
                            val_loss += loss.item() * inputs.size(0)
                    
                    val_loss /= len(val_loader.dataset)
                    
                    # Update training stats
                    self.trainer.training_stats['train_losses'].append(train_loss)
                    self.trainer.training_stats['val_losses'].append(val_loss)
                    self.trainer.training_stats['epochs'] = epoch + 1
                    
                    # Emit progress
                    self.epoch_completed.emit(epoch+1, train_loss, val_loss)
                    
                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_epochs = 0
                        self.trainer.save_model(os.path.join(self.trainer.model_dir, 'best_facial_model.pth'))
                    else:
                        no_improve_epochs += 1
                        
                    # Early stopping
                    if no_improve_epochs >= kwargs.get('patience', 10):
                        self.progress_updated.emit(100, f"Early stopping after {epoch+1} epochs")
                        break
                
                return self.trainer.training_stats
            
            # Temporarily replace train method
            self.trainer.train = train_with_progress
            
            # Start training
            stats = self.trainer.train(
                self.dataset, 
                val_split=0.2,
                batch_size=self.batch_size, 
                epochs=self.epochs, 
                patience=self.patience
            )
            
            # Signal completion
            self.progress_updated.emit(100, "Training completed!")
            self.training_completed.emit(stats)
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.training_error.emit(f"Training error: {str(e)}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Training metrics visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax1 = self.figure.add_subplot(111)
        self.train_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax1.plot([], [], 'r-', label='Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Initialize data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_progress(self, progress: int, status: str):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def update_plot(self, epoch: int, train_loss: float, val_loss: float):
        """Update the training plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits
        if len(self.epochs) > 0:
            self.ax1.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                self.ax1.set_ylim(0, max(all_losses) * 1.1)
        
        self.canvas.draw()
        
    def reset(self):
        """Reset the visualizer to initial state."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 1)
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")

class TrainingManager:
    """Manager for the facial feature model training process."""
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """Initialize the training manager.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trainer = FacialFeatureTrainer(model_dir=self.model_dir)
        self.dataset = None
        self.worker = None
        self.visualizer = None
        
    def load_dataset(self, use_all_landmarks: bool = True) -> int:
        """Load the training dataset.
        
        Args:
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
            
        Returns:
            int: Number of samples loaded
        """
        self.dataset = FacialFeatureDataset(self.data_dir, use_all_landmarks=use_all_landmarks)
        return len(self.dataset)
    
    def create_visualizer(self, parent=None) -> TrainingVisualizer:
        """Create a training visualizer widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            TrainingVisualizer: The created visualizer widget
        """
        self.visualizer = TrainingVisualizer(parent)
        return self.visualizer
    
    def start_training(self, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> bool:
        """Start the training process.
        
        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            bool: True if training started successfully
        """
        if self.worker and self.worker.is_running:
            logger.warning("Training is already in progress")
            return False
        
        # Load dataset if not already loaded
        if not self.dataset or len(self.dataset) == 0:
            try:
                samples = self.load_dataset()
                if samples == 0:
                    logger.error("No training data available")
                    return False
                logger.info(f"Loaded {samples} training samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {str(e)}")
                return False
        
        # Validate dataset before proceeding
        if len(self.dataset.samples) == 0:
            logger.error("Dataset is empty after loading")
            return False
        
        # Create and configure worker
        try:
            self.worker = TrainingWorker(
                trainer=self.trainer,
                dataset=self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Connect signals if visualizer exists
            if self.visualizer:
                self.worker.progress_updated.connect(self.visualizer.update_progress)
                self.worker.epoch_completed.connect(self.visualizer.update_plot)
                self.visualizer.reset()
            
            # Start training
            self.worker.start()
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def stop_training(self) -> bool:
        """Stop the training process.
        
        Returns:
            bool: True if training was stopped
        """
        if self.worker and self.worker.is_running:
            self.worker.stop()
            return True
        return False
    
    def is_training_in_progress(self) -> bool:
        """Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress
        """
        return self.worker is not None and self.worker.is_running
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dict: Model information including training stats
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'not_trained',
                'message': 'No model has been trained yet'
            }
        
        return {
            'status': 'trained',
            'epochs': self.trainer.training_stats.get('epochs', 0),
            'final_train_loss': self.trainer.training_stats.get('train_losses', [])[-1] if self.trainer.training_stats.get('train_losses') else None,
            'final_val_loss': self.trainer.training_stats.get('val_losses', [])[-1] if self.trainer.training_stats.get('val_losses') else None,
            'model_path': os.path.join(self.model_dir, 'final_facial_model.pth')
        }
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks
            
        Returns:
            Dict: Prediction results
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'error',
                'message': 'No trained model available'
            }
        
        try:
            # Make prediction using trainer's model
            predictions = self.trainer.predict(landmarks)
            
            # Check if we're using the landmark model (output will be large)
            is_landmark_model = hasattr(self.trainer.model, 'is_landmark_model') and self.trainer.model.is_landmark_model
            
            if is_landmark_model and isinstance(predictions, np.ndarray) and predictions.size > 10:
                # For landmark model, return the full landmark predictions
                # Convert to a serializable format
                landmarks_dict = {}
                for i, landmark in enumerate(predictions):
                    landmarks_dict[f'landmark_{i}'] = {
                        'x': float(landmark[0]),
                        'y': float(landmark[1]),
                        'z': float(landmark[2]) if landmark.size > 2 else 0.0
                    }
                
                return {
                    'status': 'success',
                    'model_type': 'landmark',
                    'predictions': landmarks_dict
                }
            else:
                # For the original metrics model
                return {
                    'status': 'success',
                    'model_type': 'metrics',
                    'predictions': {
                        'left_eye_ratio': float(predictions[0]),
                        'right_eye_ratio': float(predictions[1]),
                        'mouth_aspect_ratio': float(predictions[2]),
                        'left_brow_height': float(predictions[3]),
                        'right_brow_height': float(predictions[4])
                    }
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Training metrics visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax1 = self.figure.add_subplot(111)
        self.train_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax1.plot([], [], 'r-', label='Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Initialize data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_progress(self, progress: int, status: str):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def update_plot(self, epoch: int, train_loss: float, val_loss: float):
        """Update the training plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits
        if len(self.epochs) > 0:
            self.ax1.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                self.ax1.set_ylim(0, max(all_losses) * 1.1)
        
        self.canvas.draw()
        
    def reset(self):
        """Reset the visualizer to initial state."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 1)
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")

class TrainingManager:
    """Manager for the facial feature model training process."""
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """Initialize the training manager.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trainer = FacialFeatureTrainer(model_dir=self.model_dir)
        self.dataset = None
        self.worker = None
        self.visualizer = None
        
    def load_dataset(self, use_all_landmarks: bool = True) -> int:
        """Load the training dataset.
        
        Args:
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
            
        Returns:
            int: Number of samples loaded
        """
        self.dataset = FacialFeatureDataset(self.data_dir, use_all_landmarks=use_all_landmarks)
        return len(self.dataset)
    
    def create_visualizer(self, parent=None) -> TrainingVisualizer:
        """Create a training visualizer widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            TrainingVisualizer: The created visualizer widget
        """
        self.visualizer = TrainingVisualizer(parent)
        return self.visualizer
    
    def start_training(self, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> bool:
        """Start the training process.
        
        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            bool: True if training started successfully
        """
        if self.worker and self.worker.is_running:
            logger.warning("Training is already in progress")
            return False
        
        # Load dataset if not already loaded
        if not self.dataset or len(self.dataset) == 0:
            try:
                samples = self.load_dataset()
                if samples == 0:
                    logger.error("No training data available")
                    return False
                logger.info(f"Loaded {samples} training samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {str(e)}")
                return False
        
        # Validate dataset before proceeding
        if len(self.dataset.samples) == 0:
            logger.error("Dataset is empty after loading")
            return False
        
        # Create and configure worker
        try:
            self.worker = TrainingWorker(
                trainer=self.trainer,
                dataset=self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Connect signals if visualizer exists
            if self.visualizer:
                self.worker.progress_updated.connect(self.visualizer.update_progress)
                self.worker.epoch_completed.connect(self.visualizer.update_plot)
                self.visualizer.reset()
            
            # Start training
            self.worker.start()
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def stop_training(self) -> bool:
        """Stop the training process.
        
        Returns:
            bool: True if training was stopped
        """
        if self.worker and self.worker.is_running:
            self.worker.stop()
            return True
        return False
    
    def is_training_in_progress(self) -> bool:
        """Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress
        """
        return self.worker is not None and self.worker.is_running
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dict: Model information including training stats
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'not_trained',
                'message': 'No model has been trained yet'
            }
        
        return {
            'status': 'trained',
            'epochs': self.trainer.training_stats.get('epochs', 0),
            'final_train_loss': self.trainer.training_stats.get('train_losses', [])[-1] if self.trainer.training_stats.get('train_losses') else None,
            'final_val_loss': self.trainer.training_stats.get('val_losses', [])[-1] if self.trainer.training_stats.get('val_losses') else None,
            'model_path': os.path.join(self.model_dir, 'final_facial_model.pth')
        }
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks
            
        Returns:
            Dict: Prediction results
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'error',
                'message': 'No trained model available'
            }
        
        try:
            # Make prediction using trainer's model
            predictions = self.trainer.predict(landmarks)
            
            # Check if we're using the landmark model (output will be large)
            is_landmark_model = hasattr(self.trainer.model, 'is_landmark_model') and self.trainer.model.is_landmark_model
            
            if is_landmark_model and isinstance(predictions, np.ndarray) and predictions.size > 10:
                # For landmark model, return the full landmark predictions
                # Convert to a serializable format
                landmarks_dict = {}
                for i, landmark in enumerate(predictions):
                    landmarks_dict[f'landmark_{i}'] = {
                        'x': float(landmark[0]),
                        'y': float(landmark[1]),
                        'z': float(landmark[2]) if landmark.size > 2 else 0.0
                    }
                
                return {
                    'status': 'success',
                    'model_type': 'landmark',
                    'predictions': landmarks_dict
                }
            else:
                # For the original metrics model
                return {
                    'status': 'success',
                    'model_type': 'metrics',
                    'predictions': {
                        'left_eye_ratio': float(predictions[0]),
                        'right_eye_ratio': float(predictions[1]),
                        'mouth_aspect_ratio': float(predictions[2]),
                        'left_brow_height': float(predictions[3]),
                        'right_brow_height': float(predictions[4])
                    }
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Training metrics visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax1 = self.figure.add_subplot(111)
        self.train_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax1.plot([], [], 'r-', label='Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Initialize data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_progress(self, progress: int, status: str):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def update_plot(self, epoch: int, train_loss: float, val_loss: float):
        """Update the training plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits
        if len(self.epochs) > 0:
            self.ax1.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                self.ax1.set_ylim(0, max(all_losses) * 1.1)
        
        self.canvas.draw()
        
    def reset(self):
        """Reset the visualizer to initial state."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 1)
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")

class TrainingManager:
    """Manager for the facial feature model training process."""
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """Initialize the training manager.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trainer = FacialFeatureTrainer(model_dir=self.model_dir)
        self.dataset = None
        self.worker = None
        self.visualizer = None
        
    def load_dataset(self, use_all_landmarks: bool = True) -> int:
        """Load the training dataset.
        
        Args:
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
            
        Returns:
            int: Number of samples loaded
        """
        self.dataset = FacialFeatureDataset(self.data_dir, use_all_landmarks=use_all_landmarks)
        return len(self.dataset)
    
    def create_visualizer(self, parent=None) -> TrainingVisualizer:
        """Create a training visualizer widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            TrainingVisualizer: The created visualizer widget
        """
        self.visualizer = TrainingVisualizer(parent)
        return self.visualizer
    
    def start_training(self, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> bool:
        """Start the training process.
        
        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            bool: True if training started successfully
        """
        if self.worker and self.worker.is_running:
            logger.warning("Training is already in progress")
            return False
        
        # Load dataset if not already loaded
        if not self.dataset or len(self.dataset) == 0:
            try:
                samples = self.load_dataset()
                if samples == 0:
                    logger.error("No training data available")
                    return False
                logger.info(f"Loaded {samples} training samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {str(e)}")
                return False
        
        # Validate dataset before proceeding
        if len(self.dataset.samples) == 0:
            logger.error("Dataset is empty after loading")
            return False
        
        # Create and configure worker
        try:
            self.worker = TrainingWorker(
                trainer=self.trainer,
                dataset=self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Connect signals if visualizer exists
            if self.visualizer:
                self.worker.progress_updated.connect(self.visualizer.update_progress)
                self.worker.epoch_completed.connect(self.visualizer.update_plot)
                self.visualizer.reset()
            
            # Start training
            self.worker.start()
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def stop_training(self) -> bool:
        """Stop the training process.
        
        Returns:
            bool: True if training was stopped
        """
        if self.worker and self.worker.is_running:
            self.worker.stop()
            return True
        return False
    
    def is_training_in_progress(self) -> bool:
        """Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress
        """
        return self.worker is not None and self.worker.is_running
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dict: Model information including training stats
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'not_trained',
                'message': 'No model has been trained yet'
            }
        
        return {
            'status': 'trained',
            'epochs': self.trainer.training_stats.get('epochs', 0),
            'final_train_loss': self.trainer.training_stats.get('train_losses', [])[-1] if self.trainer.training_stats.get('train_losses') else None,
            'final_val_loss': self.trainer.training_stats.get('val_losses', [])[-1] if self.trainer.training_stats.get('val_losses') else None,
            'model_path': os.path.join(self.model_dir, 'final_facial_model.pth')
        }
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks
            
        Returns:
            Dict: Prediction results
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'error',
                'message': 'No trained model available'
            }
        
        try:
            # Make prediction using trainer's model
            predictions = self.trainer.predict(landmarks)
            
            # Check if we're using the landmark model (output will be large)
            is_landmark_model = hasattr(self.trainer.model, 'is_landmark_model') and self.trainer.model.is_landmark_model
            
            if is_landmark_model and isinstance(predictions, np.ndarray) and predictions.size > 10:
                # For landmark model, return the full landmark predictions
                # Convert to a serializable format
                landmarks_dict = {}
                for i, landmark in enumerate(predictions):
                    landmarks_dict[f'landmark_{i}'] = {
                        'x': float(landmark[0]),
                        'y': float(landmark[1]),
                        'z': float(landmark[2]) if landmark.size > 2 else 0.0
                    }
                
                return {
                    'status': 'success',
                    'model_type': 'landmark',
                    'predictions': landmarks_dict
                }
            else:
                # For the original metrics model
                return {
                    'status': 'success',
                    'model_type': 'metrics',
                    'predictions': {
                        'left_eye_ratio': float(predictions[0]),
                        'right_eye_ratio': float(predictions[1]),
                        'mouth_aspect_ratio': float(predictions[2]),
                        'left_brow_height': float(predictions[3]),
                        'right_brow_height': float(predictions[4])
                    }
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Training metrics visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax1 = self.figure.add_subplot(111)
        self.train_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax1.plot([], [], 'r-', label='Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Initialize data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_progress(self, progress: int, status: str):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def update_plot(self, epoch: int, train_loss: float, val_loss: float):
        """Update the training plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits
        if len(self.epochs) > 0:
            self.ax1.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                self.ax1.set_ylim(0, max(all_losses) * 1.1)
        
        self.canvas.draw()
        
    def reset(self):
        """Reset the visualizer to initial state."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 1)
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")

class TrainingManager:
    """Manager for the facial feature model training process."""
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """Initialize the training manager.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trainer = FacialFeatureTrainer(model_dir=self.model_dir)
        self.dataset = None
        self.worker = None
        self.visualizer = None
        
    def load_dataset(self, use_all_landmarks: bool = True) -> int:
        """Load the training dataset.
        
        Args:
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
            
        Returns:
            int: Number of samples loaded
        """
        self.dataset = FacialFeatureDataset(self.data_dir, use_all_landmarks=use_all_landmarks)
        return len(self.dataset)
    
    def create_visualizer(self, parent=None) -> TrainingVisualizer:
        """Create a training visualizer widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            TrainingVisualizer: The created visualizer widget
        """
        self.visualizer = TrainingVisualizer(parent)
        return self.visualizer
    
    def start_training(self, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> bool:
        """Start the training process.
        
        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            bool: True if training started successfully
        """
        if self.worker and self.worker.is_running:
            logger.warning("Training is already in progress")
            return False
        
        # Load dataset if not already loaded
        if not self.dataset or len(self.dataset) == 0:
            try:
                samples = self.load_dataset()
                if samples == 0:
                    logger.error("No training data available")
                    return False
                logger.info(f"Loaded {samples} training samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {str(e)}")
                return False
        
        # Validate dataset before proceeding
        if len(self.dataset.samples) == 0:
            logger.error("Dataset is empty after loading")
            return False
        
        # Create and configure worker
        try:
            self.worker = TrainingWorker(
                trainer=self.trainer,
                dataset=self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Connect signals if visualizer exists
            if self.visualizer:
                self.worker.progress_updated.connect(self.visualizer.update_progress)
                self.worker.epoch_completed.connect(self.visualizer.update_plot)
                self.visualizer.reset()
            
            # Start training
            self.worker.start()
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def stop_training(self) -> bool:
        """Stop the training process.
        
        Returns:
            bool: True if training was stopped
        """
        if self.worker and self.worker.is_running:
            self.worker.stop()
            return True
        return False
    
    def is_training_in_progress(self) -> bool:
        """Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress
        """
        return self.worker is not None and self.worker.is_running
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dict: Model information including training stats
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'not_trained',
                'message': 'No model has been trained yet'
            }
        
        return {
            'status': 'trained',
            'epochs': self.trainer.training_stats.get('epochs', 0),
            'final_train_loss': self.trainer.training_stats.get('train_losses', [])[-1] if self.trainer.training_stats.get('train_losses') else None,
            'final_val_loss': self.trainer.training_stats.get('val_losses', [])[-1] if self.trainer.training_stats.get('val_losses') else None,
            'model_path': os.path.join(self.model_dir, 'final_facial_model.pth')
        }
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks
            
        Returns:
            Dict: Prediction results
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'error',
                'message': 'No trained model available'
            }
        
        try:
            # Make prediction using trainer's model
            predictions = self.trainer.predict(landmarks)
            
            # Check if we're using the landmark model (output will be large)
            is_landmark_model = hasattr(self.trainer.model, 'is_landmark_model') and self.trainer.model.is_landmark_model
            
            if is_landmark_model and isinstance(predictions, np.ndarray) and predictions.size > 10:
                # For landmark model, return the full landmark predictions
                # Convert to a serializable format
                landmarks_dict = {}
                for i, landmark in enumerate(predictions):
                    landmarks_dict[f'landmark_{i}'] = {
                        'x': float(landmark[0]),
                        'y': float(landmark[1]),
                        'z': float(landmark[2]) if landmark.size > 2 else 0.0
                    }
                
                return {
                    'status': 'success',
                    'model_type': 'landmark',
                    'predictions': landmarks_dict
                }
            else:
                # For the original metrics model
                return {
                    'status': 'success',
                    'model_type': 'metrics',
                    'predictions': {
                        'left_eye_ratio': float(predictions[0]),
                        'right_eye_ratio': float(predictions[1]),
                        'mouth_aspect_ratio': float(predictions[2]),
                        'left_brow_height': float(predictions[3]),
                        'right_brow_height': float(predictions[4])
                    }
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Training metrics visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax1 = self.figure.add_subplot(111)
        self.train_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax1.plot([], [], 'r-', label='Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Initialize data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_progress(self, progress: int, status: str):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def update_plot(self, epoch: int, train_loss: float, val_loss: float):
        """Update the training plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits
        if len(self.epochs) > 0:
            self.ax1.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                self.ax1.set_ylim(0, max(all_losses) * 1.1)
        
        self.canvas.draw()
        
    def reset(self):
        """Reset the visualizer to initial state."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 1)
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")

class TrainingManager:
    """Manager for the facial feature model training process."""
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """Initialize the training manager.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trainer = FacialFeatureTrainer(model_dir=self.model_dir)
        self.dataset = None
        self.worker = None
        self.visualizer = None
        
    def load_dataset(self, use_all_landmarks: bool = True) -> int:
        """Load the training dataset.
        
        Args:
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
            
        Returns:
            int: Number of samples loaded
        """
        self.dataset = FacialFeatureDataset(self.data_dir, use_all_landmarks=use_all_landmarks)
        return len(self.dataset)
    
    def create_visualizer(self, parent=None) -> TrainingVisualizer:
        """Create a training visualizer widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            TrainingVisualizer: The created visualizer widget
        """
        self.visualizer = TrainingVisualizer(parent)
        return self.visualizer
    
    def start_training(self, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> bool:
        """Start the training process.
        
        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            bool: True if training started successfully
        """
        if self.worker and self.worker.is_running:
            logger.warning("Training is already in progress")
            return False
        
        # Load dataset if not already loaded
        if not self.dataset or len(self.dataset) == 0:
            try:
                samples = self.load_dataset()
                if samples == 0:
                    logger.error("No training data available")
                    return False
                logger.info(f"Loaded {samples} training samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {str(e)}")
                return False
        
        # Validate dataset before proceeding
        if len(self.dataset.samples) == 0:
            logger.error("Dataset is empty after loading")
            return False
        
        # Create and configure worker
        try:
            self.worker = TrainingWorker(
                trainer=self.trainer,
                dataset=self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Connect signals if visualizer exists
            if self.visualizer:
                self.worker.progress_updated.connect(self.visualizer.update_progress)
                self.worker.epoch_completed.connect(self.visualizer.update_plot)
                self.visualizer.reset()
            
            # Start training
            self.worker.start()
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def stop_training(self) -> bool:
        """Stop the training process.
        
        Returns:
            bool: True if training was stopped
        """
        if self.worker and self.worker.is_running:
            self.worker.stop()
            return True
        return False
    
    def is_training_in_progress(self) -> bool:
        """Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress
        """
        return self.worker is not None and self.worker.is_running
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dict: Model information including training stats
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'not_trained',
                'message': 'No model has been trained yet'
            }
        
        return {
            'status': 'trained',
            'epochs': self.trainer.training_stats.get('epochs', 0),
            'final_train_loss': self.trainer.training_stats.get('train_losses', [])[-1] if self.trainer.training_stats.get('train_losses') else None,
            'final_val_loss': self.trainer.training_stats.get('val_losses', [])[-1] if self.trainer.training_stats.get('val_losses') else None,
            'model_path': os.path.join(self.model_dir, 'final_facial_model.pth')
        }
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks
            
        Returns:
            Dict: Prediction results
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'error',
                'message': 'No trained model available'
            }
        
        try:
            # Make prediction using trainer's model
            predictions = self.trainer.predict(landmarks)
            
            # Check if we're using the landmark model (output will be large)
            is_landmark_model = hasattr(self.trainer.model, 'is_landmark_model') and self.trainer.model.is_landmark_model
            
            if is_landmark_model and isinstance(predictions, np.ndarray) and predictions.size > 10:
                # For landmark model, return the full landmark predictions
                # Convert to a serializable format
                landmarks_dict = {}
                for i, landmark in enumerate(predictions):
                    landmarks_dict[f'landmark_{i}'] = {
                        'x': float(landmark[0]),
                        'y': float(landmark[1]),
                        'z': float(landmark[2]) if landmark.size > 2 else 0.0
                    }
                
                return {
                    'status': 'success',
                    'model_type': 'landmark',
                    'predictions': landmarks_dict
                }
            else:
                # For the original metrics model
                return {
                    'status': 'success',
                    'model_type': 'metrics',
                    'predictions': {
                        'left_eye_ratio': float(predictions[0]),
                        'right_eye_ratio': float(predictions[1]),
                        'mouth_aspect_ratio': float(predictions[2]),
                        'left_brow_height': float(predictions[3]),
                        'right_brow_height': float(predictions[4])
                    }
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Training metrics visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax1 = self.figure.add_subplot(111)
        self.train_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax1.plot([], [], 'r-', label='Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Initialize data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_progress(self, progress: int, status: str):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def update_plot(self, epoch: int, train_loss: float, val_loss: float):
        """Update the training plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits
        if len(self.epochs) > 0:
            self.ax1.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                self.ax1.set_ylim(0, max(all_losses) * 1.1)
        
        self.canvas.draw()
        
    def reset(self):
        """Reset the visualizer to initial state."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 1)
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")

class TrainingManager:
    """Manager for the facial feature model training process."""
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """Initialize the training manager.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trainer = FacialFeatureTrainer(model_dir=self.model_dir)
        self.dataset = None
        self.worker = None
        self.visualizer = None
        
    def load_dataset(self, use_all_landmarks: bool = True) -> int:
        """Load the training dataset.
        
        Args:
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
            
        Returns:
            int: Number of samples loaded
        """
        self.dataset = FacialFeatureDataset(self.data_dir, use_all_landmarks=use_all_landmarks)
        return len(self.dataset)
    
    def create_visualizer(self, parent=None) -> TrainingVisualizer:
        """Create a training visualizer widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            TrainingVisualizer: The created visualizer widget
        """
        self.visualizer = TrainingVisualizer(parent)
        return self.visualizer
    
    def start_training(self, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> bool:
        """Start the training process.
        
        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            bool: True if training started successfully
        """
        if self.worker and self.worker.is_running:
            logger.warning("Training is already in progress")
            return False
        
        # Load dataset if not already loaded
        if not self.dataset or len(self.dataset) == 0:
            try:
                samples = self.load_dataset()
                if samples == 0:
                    logger.error("No training data available")
                    return False
                logger.info(f"Loaded {samples} training samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {str(e)}")
                return False
        
        # Validate dataset before proceeding
        if len(self.dataset.samples) == 0:
            logger.error("Dataset is empty after loading")
            return False
        
        # Create and configure worker
        try:
            self.worker = TrainingWorker(
                trainer=self.trainer,
                dataset=self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Connect signals if visualizer exists
            if self.visualizer:
                self.worker.progress_updated.connect(self.visualizer.update_progress)
                self.worker.epoch_completed.connect(self.visualizer.update_plot)
                self.visualizer.reset()
            
            # Start training
            self.worker.start()
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def stop_training(self) -> bool:
        """Stop the training process.
        
        Returns:
            bool: True if training was stopped
        """
        if self.worker and self.worker.is_running:
            self.worker.stop()
            return True
        return False
    
    def is_training_in_progress(self) -> bool:
        """Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress
        """
        return self.worker is not None and self.worker.is_running
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dict: Model information including training stats
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'not_trained',
                'message': 'No model has been trained yet'
            }
        
        return {
            'status': 'trained',
            'epochs': self.trainer.training_stats.get('epochs', 0),
            'final_train_loss': self.trainer.training_stats.get('train_losses', [])[-1] if self.trainer.training_stats.get('train_losses') else None,
            'final_val_loss': self.trainer.training_stats.get('val_losses', [])[-1] if self.trainer.training_stats.get('val_losses') else None,
            'model_path': os.path.join(self.model_dir, 'final_facial_model.pth')
        }
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks
            
        Returns:
            Dict: Prediction results
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'error',
                'message': 'No trained model available'
            }
        
        try:
            # Make prediction using trainer's model
            predictions = self.trainer.predict(landmarks)
            
            # Check if we're using the landmark model (output will be large)
            is_landmark_model = hasattr(self.trainer.model, 'is_landmark_model') and self.trainer.model.is_landmark_model
            
            if is_landmark_model and isinstance(predictions, np.ndarray) and predictions.size > 10:
                # For landmark model, return the full landmark predictions
                # Convert to a serializable format
                landmarks_dict = {}
                for i, landmark in enumerate(predictions):
                    landmarks_dict[f'landmark_{i}'] = {
                        'x': float(landmark[0]),
                        'y': float(landmark[1]),
                        'z': float(landmark[2]) if landmark.size > 2 else 0.0
                    }
                
                return {
                    'status': 'success',
                    'model_type': 'landmark',
                    'predictions': landmarks_dict
                }
            else:
                # For the original metrics model
                return {
                    'status': 'success',
                    'model_type': 'metrics',
                    'predictions': {
                        'left_eye_ratio': float(predictions[0]),
                        'right_eye_ratio': float(predictions[1]),
                        'mouth_aspect_ratio': float(predictions[2]),
                        'left_brow_height': float(predictions[3]),
                        'right_brow_height': float(predictions[4])
                    }
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Training metrics visualization
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize plots
        self.ax1 = self.figure.add_subplot(111)
        self.train_line, = self.ax1.plot([], [], 'b-', label='Training Loss')
        self.val_line, = self.ax1.plot([], [], 'r-', label='Validation Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_title('Training Progress')
        self.ax1.legend()
        self.ax1.grid(True)
        
        # Initialize data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def update_progress(self, progress: int, status: str):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        
    def update_plot(self, epoch: int, train_loss: float, val_loss: float):
        """Update the training plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        
        # Adjust plot limits
        if len(self.epochs) > 0:
            self.ax1.set_xlim(0, max(self.epochs) + 1)
            all_losses = self.train_losses + self.val_losses
            if all_losses:
                self.ax1.set_ylim(0, max(all_losses) * 1.1)
        
        self.canvas.draw()
        
    def reset(self):
        """Reset the visualizer to initial state."""
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.train_line.set_data([], [])
        self.val_line.set_data([], [])
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(0, 1)
        self.canvas.draw()
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready to train")

class TrainingManager:
    """Manager for the facial feature model training process."""
    
    def __init__(self, data_dir: str = None, model_dir: str = None):
        """Initialize the training manager.
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory to save trained models
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'training_data')
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.trainer = FacialFeatureTrainer(model_dir=self.model_dir)
        self.dataset = None
        self.worker = None
        self.visualizer = None
        
    def load_dataset(self, use_all_landmarks: bool = True) -> int:
        """Load the training dataset.
        
        Args:
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
            
        Returns:
            int: Number of samples loaded
        """
        self.dataset = FacialFeatureDataset(self.data_dir, use_all_landmarks=use_all_landmarks)
        return len(self.dataset)
    
    def create_visualizer(self, parent=None) -> TrainingVisualizer:
        """Create a training visualizer widget.
        
        Args:
            parent: Parent widget
            
        Returns:
            TrainingVisualizer: The created visualizer widget
        """
        self.visualizer = TrainingVisualizer(parent)
        return self.visualizer
    
    def start_training(self, epochs: int = 50, batch_size: int = 32, patience: int = 10) -> bool:
        """Start the training process.
        
        Args:
            epochs: Maximum number of epochs to train
            batch_size: Batch size for training
            patience: Number of epochs to wait for improvement before early stopping
            
        Returns:
            bool: True if training started successfully
        """
        if self.worker and self.worker.is_running:
            logger.warning("Training is already in progress")
            return False
        
        # Load dataset if not already loaded
        if not self.dataset or len(self.dataset) == 0:
            try:
                samples = self.load_dataset()
                if samples == 0:
                    logger.error("No training data available")
                    return False
                logger.info(f"Loaded {samples} training samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {str(e)}")
                return False
        
        # Validate dataset before proceeding
        if len(self.dataset.samples) == 0:
            logger.error("Dataset is empty after loading")
            return False
        
        # Create and configure worker
        try:
            self.worker = TrainingWorker(
                trainer=self.trainer,
                dataset=self.dataset,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Connect signals if visualizer exists
            if self.visualizer:
                self.worker.progress_updated.connect(self.visualizer.update_progress)
                self.worker.epoch_completed.connect(self.visualizer.update_plot)
                self.visualizer.reset()
            
            # Start training
            self.worker.start()
            logger.info("Training started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training: {str(e)}")
            return False
    
    def stop_training(self) -> bool:
        """Stop the training process.
        
        Returns:
            bool: True if training was stopped
        """
        if self.worker and self.worker.is_running:
            self.worker.stop()
            return True
        return False
    
    def is_training_in_progress(self) -> bool:
        """Check if training is currently in progress.
        
        Returns:
            bool: True if training is in progress
        """
        return self.worker is not None and self.worker.is_running
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model.
        
        Returns:
            Dict: Model information including training stats
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'not_trained',
                'message': 'No model has been trained yet'
            }
        
        return {
            'status': 'trained',
            'epochs': self.trainer.training_stats.get('epochs', 0),
            'final_train_loss': self.trainer.training_stats.get('train_losses', [])[-1] if self.trainer.training_stats.get('train_losses') else None,
            'final_val_loss': self.trainer.training_stats.get('val_losses', [])[-1] if self.trainer.training_stats.get('val_losses') else None,
            'model_path': os.path.join(self.model_dir, 'final_facial_model.pth')
        }
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks
            
        Returns:
            Dict: Prediction results
        """
        if not self.trainer or not self.trainer.model:
            return {
                'status': 'error',
                'message': 'No trained model available'
            }
        
        try:
            # Make prediction using trainer's model
            predictions = self.trainer.predict(landmarks)
            
            # Check if we're using the landmark model (output will be large)
            is_landmark_model = hasattr(self.trainer.model, 'is_landmark_model') and self.trainer.model.is_landmark_model
            
            if is_landmark_model and isinstance(predictions, np.ndarray) and predictions.size > 10:
                # For landmark model, return the full landmark predictions
                # Convert to a serializable format
                landmarks_dict = {}
                for i, landmark in enumerate(predictions):
                    landmarks_dict[f'landmark_{i}'] = {
                        'x': float(landmark[0]),
                        'y': float(landmark[1]),
                        'z': float(landmark[2]) if landmark.size > 2 else 0.0
                    }
                
                return {
                    'status': 'success',
                    'model_type': 'landmark',
                    'predictions': landmarks_dict
                }
            else:
                # For the original metrics model
                return {
                    'status': 'success',
                    'model_type': 'metrics',
                    'predictions': {
                        'left_eye_ratio': float(predictions[0]),
                        'right_eye_ratio': float(predictions[1]),
                        'mouth_aspect_ratio': float(predictions[2]),
                        'left_brow_height': float(predictions[3]),
                        'right_brow_height': float(predictions[4])
                    }
                }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to make prediction: {str(e)}'
            }
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False
        self.terminate()
        self.wait()

class TrainingVisualizer(QWidget):
    """Widget for visualizing the training process."""
    
    def __init__(self, parent=None):
        """Initialize the training visualizer widget."""
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Ready to train")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_