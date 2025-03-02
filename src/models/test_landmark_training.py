#!/usr/bin/env python3

"""
Test script for training the facial feature model on all 468 landmarks.

This script demonstrates training the model on all individual landmark points
that MediaPipe provides, aiming for minimal loss.
"""

import os
import sys
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.facial_feature_model import FacialFeatureDataset, FacialFeatureTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the training test."""
    # Set paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'training_data')
    model_dir = os.path.join(base_dir, 'models')
    
    # Ensure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if training data exists
    if not os.path.exists(data_dir):
        logger.error(f"Training data directory not found: {data_dir}")
        logger.info("Please run the training capture process first to collect facial landmark data.")
        return
    
    # Load dataset with all landmarks
    logger.info("Loading dataset with all 468 landmarks...")
    try:
        dataset = FacialFeatureDataset(data_dir, use_all_landmarks=True)
        logger.info(f"Loaded {len(dataset)} training samples")
        
        if len(dataset) == 0:
            logger.error("No training samples found. Please capture training data first.")
            return
            
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return
    
    # Create trainer
    logger.info("Creating facial feature trainer...")
    trainer = FacialFeatureTrainer(model_dir=model_dir)
    
    # Configure training parameters for minimal loss
    epochs = 1000  # High number of epochs
    batch_size = 16  # Smaller batch size for better convergence
    patience = 50   # More patience to allow the model to converge further
    learning_rate = 0.0005  # Lower learning rate for finer convergence
    
    # Get a sample to determine input/output sizes
    sample = dataset[0]
    input_size = sample['input'].shape[0]
    output_size = sample['target'].shape[0]
    
    logger.info(f"Input size: {input_size}, Output size: {output_size}")
    
    # Create model with larger hidden size for the complex task
    trainer.create_model(input_size, hidden_size=2048, output_size=output_size)
    
    # Use Adam optimizer with custom learning rate
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=learning_rate)
    
    # Train the model
    logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}, patience {patience}...")
    start_time = datetime.now()
    
    try:
        stats = trainer.train(dataset, val_split=0.1, batch_size=batch_size, epochs=epochs, patience=patience)
        
        # Training completed
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in {training_time}")
        
        # Report final losses
        final_train_loss = stats['train_losses'][-1]
        final_val_loss = stats['val_losses'][-1]
        logger.info(f"Final training loss: {final_train_loss:.8f}")
        logger.info(f"Final validation loss: {final_val_loss:.8f}")
        
        # Plot training curves
        plt.figure(figsize=(10, 6))
        plt.plot(stats['train_losses'], label='Training Loss')
        plt.plot(stats['val_losses'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(model_dir, 'landmark_training_curves.png')
        plt.savefig(plot_path)
        logger.info(f"Training curves saved to {plot_path}")
        
        # Save the model
        model_path = os.path.join(model_dir, 'landmark_model.pth')
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()