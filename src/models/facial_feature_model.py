"""Facial Feature Model for Peeka.

This module implements a deep learning model for facial feature extraction,
expression recognition, and avatar control based on captured training data.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import cv2
import glob
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FacialFeatureDataset(Dataset):
    """Dataset for facial feature landmarks and metrics."""
    
    def __init__(self, data_dir: str, transform=None, use_all_landmarks: bool = True):
        """Initialize the dataset from a directory of training sessions.
        
        Args:
            data_dir: Path to the training_data directory
            transform: Optional transforms to apply to the data
            use_all_landmarks: Whether to use all 468 landmarks as both input and target
        """
        self.data_dir = data_dir
        self.transform = transform
        self.use_all_landmarks = use_all_landmarks
        self.samples = []
        self.load_data()
        
    def load_data(self):
        """Load all landmark data from training sessions."""
        # Check if the provided directory is a session directory itself
        if os.path.exists(os.path.join(self.data_dir, "landmarks_data")):
            # This is a single session directory
            session_dirs = [self.data_dir]
            logger.info(f"Using provided session directory: {os.path.basename(self.data_dir)}")
        else:
            # Find all session directories in the parent directory that have landmarks_data
            session_dirs = [d for d in glob.glob(os.path.join(self.data_dir, "session_*")) 
                           if os.path.isdir(d) and os.path.exists(os.path.join(d, "landmarks_data"))]
            
            if not session_dirs:
                error_msg = f"No valid training sessions found in {self.data_dir}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.info(f"Found {len(session_dirs)} training sessions with landmarks data")
        
        # Load landmark data from each session
        total_frames = 0
        valid_frames = 0
        
        for session_dir in session_dirs:
            landmarks_dir = os.path.join(session_dir, "landmarks_data")
            landmark_files = sorted(glob.glob(os.path.join(landmarks_dir, "frame_*.json")))
            
            if not landmark_files:
                logger.warning(f"No landmark files found in {session_dir}")
                continue
                
            total_frames += len(landmark_files)
            logger.info(f"Processing {len(landmark_files)} frames from {os.path.basename(session_dir)}")
            
            # Load a subset of frames to avoid memory issues
            sample_rate = max(1, len(landmark_files) // 500)  # Sample at most 500 frames per session
            
            for i, landmark_file in enumerate(landmark_files):
                if i % sample_rate != 0:
                    continue
                    
                try:
                    with open(landmark_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract features and metrics
                    if not all(key in data for key in ['landmarks_normalized', 'face_metrics', 'frame_quality']):
                        logger.warning(f"Missing required data in {landmark_file}")
                        continue
                    
                    # Use normalized landmarks as input features
                    landmarks = np.array(data['landmarks_normalized'], dtype=np.float32)
                    
                    # Validate landmarks shape
                    if landmarks.shape[1] != 3:  # Should have x, y, z coordinates
                        logger.warning(f"Invalid landmarks shape in {landmark_file}")
                        continue
                    
                    if self.use_all_landmarks:
                        # Use all landmarks as both input and target for complete landmark training
                        target = landmarks.copy()  # Use the same landmarks as target
                    else:
                        # Use face metrics as target values (original behavior)
                        metrics = data['face_metrics']
                        target = np.array([
                            metrics.get('left_eye_ratio', 0),
                            metrics.get('right_eye_ratio', 0),
                            metrics.get('mouth_aspect_ratio', 0),
                            metrics.get('left_brow_height', 0),
                            metrics.get('right_brow_height', 0)
                        ], dtype=np.float32)
                    
                    # Add frame quality as weight
                    quality = data.get('frame_quality', {}).get('quality_score', 1.0)
                    
                    self.samples.append({
                        'landmarks': landmarks,
                        'target': target,
                        'quality': quality,
                        'session': os.path.basename(session_dir),
                        'frame_id': data.get('frame_id', 0)
                    })
                    valid_frames += 1
                    
                except Exception as e:
                    logger.error(f"Error loading {landmark_file}: {str(e)}")
                    continue
        
        logger.info(f"Successfully loaded {valid_frames} samples from {total_frames} total frames")
        logger.info(f"Final dataset size: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if not self.samples:
            logger.error("No samples available in the dataset")
            raise IndexError("Dataset is empty. No samples were loaded successfully.")
            
        sample = self.samples[idx]
        
        # Flatten landmarks for input to model
        landmarks_flat = sample['landmarks'].reshape(-1)
        
        # Apply transforms if any
        if self.transform:
            landmarks_flat = self.transform(landmarks_flat)
        
        return {
            'input': torch.tensor(landmarks_flat, dtype=torch.float32),
            'target': torch.tensor(sample['target'], dtype=torch.float32),
            'quality': torch.tensor(sample['quality'], dtype=torch.float32)
        }


class FacialFeatureModel(nn.Module):
    """Neural network model for facial feature analysis."""
    
    def __init__(self, input_size: int, hidden_size: int = 1024, output_size: int = 5):
        """Initialize the model architecture.
        
        Args:
            input_size: Size of input feature vector (flattened landmarks)
            hidden_size: Size of hidden layers
            output_size: Number of output features to predict
        """
        super(FacialFeatureModel, self).__init__()
        
        # Store output_size as instance variable for use in forward method
        self.output_size = output_size
        
        # Determine if we're using the model for all landmarks or just metrics
        self.is_landmark_model = output_size > 10  # If output size is large, we're training on landmarks
        
        if self.is_landmark_model:
            # Enhanced architecture for landmark-to-landmark training
            # Using deeper network with residual connections for better gradient flow
            self.input_layer = nn.Linear(input_size, hidden_size)
            self.hidden1 = nn.Linear(hidden_size, hidden_size)
            self.hidden2 = nn.Linear(hidden_size, hidden_size)
            self.hidden3 = nn.Linear(hidden_size, hidden_size // 2)
            # Adjust output layer to match the 3D coordinates (468 landmarks * 3 coordinates)
            self.output_layer = nn.Linear(hidden_size // 2, output_size * 3)
            
            # Activation and regularization
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.batch_norm1 = nn.BatchNorm1d(hidden_size)
            self.batch_norm2 = nn.BatchNorm1d(hidden_size)
            self.batch_norm3 = nn.BatchNorm1d(hidden_size // 2)
        else:
            # Original simpler architecture for metric prediction
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, output_size)
            )
    
    def forward(self, x):
        if hasattr(self, 'is_landmark_model') and self.is_landmark_model:
            # Forward pass with residual connections
            x1 = self.relu(self.input_layer(x))
            x1 = self.dropout(self.batch_norm1(x1))
            
            # First residual block
            x2 = self.relu(self.hidden1(x1))
            x2 = self.dropout(self.batch_norm2(x2))
            x2 = x2 + x1  # Residual connection
            
            # Second residual block
            x3 = self.relu(self.hidden2(x2))
            x3 = self.dropout(x2 + x3)  # Residual connection with dropout
            
            # Final layers
            x4 = self.relu(self.hidden3(x3))
            x4 = self.dropout(self.batch_norm3(x4))
            
            output = self.output_layer(x4)
            # Reshape output to match the target shape (batch_size, num_landmarks, 3)
            batch_size = x4.size(0)
            return output.view(batch_size, self.output_size, 3)
        else:
            return self.model(x)


class FacialFeatureTrainer:
    """Trainer class for the facial feature model."""
    
    def __init__(self, model_dir: str = None):
        """Initialize the trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'epochs': 0
        }
    
    def create_model(self, input_size: int, hidden_size: int = 256, output_size: int = 5):
        """Create and initialize the model."""
        self.model = FacialFeatureModel(input_size, hidden_size, output_size)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, dataset: FacialFeatureDataset, val_split: float = 0.2, 
              batch_size: int = 32, epochs: int = 50, patience: int = 10):
        """Train the model with early stopping.
        
        Args:
            dataset: The dataset to train on
            val_split: Fraction of data to use for validation
            batch_size: Batch size for training
            epochs: Maximum number of epochs to train
            patience: Number of epochs to wait for improvement before early stopping
        
        Returns:
            Dict: Training statistics
        """
        # Validate dataset size
        total_samples = len(dataset)
        if total_samples < 2:  # Need at least 2 samples to split into train/val
            error_msg = f"Not enough samples for training. Found {total_samples} samples, minimum required is 2."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Split dataset into train and validation
        val_size = max(1, int(total_samples * val_split))  # Ensure at least 1 validation sample
        train_size = total_samples - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        logger.info(f"Training with {train_size} samples, validating with {val_size} samples")
        
        # Initialize model if not already created
        if self.model is None:
            # Get a sample to determine input size
            sample = dataset[0]
            input_size = sample['input'].shape[0]
            output_size = sample['target'].shape[0]
            self.create_model(input_size, hidden_size=256, output_size=output_size)
            logger.info(f"Created model with input size {input_size} and output size {output_size}")
        
        # Reset training stats
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'epochs': 0
        }
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        no_improve_epochs = 0
        start_time = datetime.now()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                qualities = batch['quality'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Weight loss by frame quality
                loss = self.criterion(outputs, targets)
                weighted_loss = (loss * qualities.view(-1, 1)).mean()
                
                # Backward pass and optimize
                weighted_loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Save statistics
            self.training_stats['train_losses'].append(train_loss)
            self.training_stats['val_losses'].append(val_loss)
            self.training_stats['epochs'] = epoch + 1
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                
                # Save the best model
                self.save_model(os.path.join(self.model_dir, 'best_facial_model.pth'))
                logger.info(f"Saved best model with validation loss: {val_loss:.6f}")
            else:
                no_improve_epochs += 1
            
            # Early stopping
            if no_improve_epochs >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Save final model
        self.save_model(os.path.join(self.model_dir, 'final_facial_model.pth'))
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.training_stats
    
    def save_model(self, path: str):
        """Save the model to disk."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str, input_size: int = None, hidden_size: int = 256, output_size: int = 5):
        """Load a model from disk."""
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return False
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Create model if not exists
            if self.model is None:
                if input_size is None:
                    logger.error("Input size must be provided when loading a model without an existing model")
                    return False
                self.create_model(input_size, hidden_size, output_size)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        # Instead of creating a figure directly, just prepare the data
        # and let the caller (likely in the main thread) create the figure
        plot_data = {
            'train_losses': self.training_stats['train_losses'],
            'val_losses': self.training_stats['val_losses'],
            'epochs': range(1, len(self.training_stats['train_losses']) + 1)
        }
        
        # Save the plot data for later use
        plot_path = os.path.join(self.model_dir, 'training_curves_data.npz')
        np.savez(plot_path, 
                 train_losses=np.array(self.training_stats['train_losses']),
                 val_losses=np.array(self.training_stats['val_losses']))
        logger.info(f"Training curves data saved to {plot_path}")
        
        return plot_data
    
    def predict(self, landmarks: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            landmarks: Normalized facial landmarks (can be 2D or 3D)
            
        Returns:
            np.ndarray: Predicted facial metrics or landmarks depending on model type
        """
        if self.model is None:
            logger.error("Model not initialized or trained")
            return np.zeros(5, dtype=np.float32)
        
        try:
            # Ensure landmarks are properly formatted
            if landmarks.ndim > 2:
                # Extract just the coordinates if we have a batch or extra dimensions
                landmarks = landmarks.reshape(landmarks.shape[0], -1)
            
            # Flatten landmarks for input to model
            landmarks_flat = landmarks.reshape(-1)
            
            # Convert to tensor and move to device
            input_tensor = torch.tensor(landmarks_flat, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Convert output tensor to numpy array
            predictions = output.cpu().numpy().squeeze()
            
            # Check if we're using the landmark model (output size will be large)
            is_landmark_model = hasattr(self.model, 'is_landmark_model') and self.model.is_landmark_model
            
            # If we're using the landmark model, reshape the output to match the original landmarks shape
            if is_landmark_model and predictions.size > 10:
                # Reshape to match the original landmarks shape (N, 3) where N is number of landmarks
                predictions = predictions.reshape(-1, 3)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return appropriate fallback based on model type
            if hasattr(self.model, 'is_landmark_model') and self.model.is_landmark_model:
                return np.zeros((468, 3), dtype=np.float32)  # Return zeros for all landmarks
            else:
                return np.zeros(5, dtype=np.float32)  # Return zeros for metrics