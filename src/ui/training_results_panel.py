"""Training Results Panel for visualizing model training process.

This module provides a UI panel for visualizing the training process,
showing metrics, and providing user control over training parameters.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QProgressBar, QPushButton, QGroupBox, QSpinBox,
                             QDoubleSpinBox, QComboBox, QTabWidget, QSplitter,
                             QCheckBox, QMessageBox, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from src.models.facial_feature_model import FacialFeatureDataset, FacialFeatureTrainer

class TrainingThread(QThread):
    """Thread for running model training without blocking the UI."""
    
    # Define signals
    progress_updated = pyqtSignal(int, str)
    epoch_completed = pyqtSignal(int, float, float)
    training_completed = pyqtSignal(dict)
    training_error = pyqtSignal(str)
    
    def __init__(self, trainer, dataset, params):
        """Initialize the training thread.
        
        Args:
            trainer: FacialFeatureTrainer instance
            dataset: FacialFeatureDataset instance
            params: Dictionary of training parameters
        """
        super().__init__()
        self.trainer = trainer
        self.dataset = dataset
        self.params = params
        self.is_running = False
        
    def run(self):
        """Run the training process in a separate thread."""
        self.is_running = True
        try:
            # Emit initial progress
            self.progress_updated.emit(0, "Preparing dataset...")
            
            # Ensure model is initialized before training
            if self.trainer.model is None:
                self.progress_updated.emit(5, "Initializing model...")
                sample = self.dataset[0]
                input_size = sample['input'].shape[0]
                output_size = sample['target'].shape[0]
                self.trainer.create_model(input_size, hidden_size=256, output_size=output_size)
            
            # Set learning rate if provided and optimizer exists
            if 'learning_rate' in self.params and self.trainer.optimizer is not None:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] = self.params['learning_rate']
            
            # Train the model using the actual trainer's train method
            stats = self.trainer.train(
                dataset=self.dataset,
                val_split=self.params.get('val_split', 0.2),
                batch_size=self.params.get('batch_size', 32),
                epochs=self.params.get('epochs', 50),
                patience=self.params.get('patience', 10)
            )
            
            # Emit completion signal
            self.training_completed.emit(stats)
            
        except Exception as e:
            self.training_error.emit(str(e))
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the training process."""
        self.is_running = False


class MetricsPlot(FigureCanvas):
    """Canvas for plotting training metrics."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Initialize the plot canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize plot data
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
        # Setup plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the initial plot."""
        self.axes.set_title('Training Progress')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.grid(True)
        self.fig.tight_layout()
        
    def update_plot(self, epoch, train_loss, val_loss):
        """Update the plot with new data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        # Add new data
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
        
        # Clear and redraw
        self.axes.clear()
        self.axes.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        self.axes.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        self.axes.legend()
        self.axes.set_title('Training Progress')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.grid(True)
        
        # Update y-axis limits for better visualization
        if len(self.train_losses) > 0:
            max_loss = max(max(self.train_losses), max(self.val_losses))
            min_loss = min(min(self.train_losses), min(self.val_losses))
            padding = (max_loss - min_loss) * 0.1
            self.axes.set_ylim([max(0, min_loss - padding), max_loss + padding])
        
        self.fig.tight_layout()
        self.draw()


class ExpressionVarietyPlot(FigureCanvas):
    """Canvas for visualizing expression variety in the dataset."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """Initialize the plot canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Setup plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the initial plot."""
        self.axes.set_title('Expression Variety')
        self.axes.set_xlabel('Feature')
        self.axes.set_ylabel('Value Range')
        self.axes.grid(True)
        self.fig.tight_layout()
        
    def update_plot(self, dataset):
        """Update the plot with dataset statistics.
        
        Args:
            dataset: FacialFeatureDataset instance
        """
        if not dataset or not hasattr(dataset, 'samples') or len(dataset.samples) == 0:
            return
            
        # Extract metrics from dataset
        metrics = ['Left Eye', 'Right Eye', 'Mouth', 'Left Brow', 'Right Brow']
        values = [[], [], [], [], []]
        
        for sample in dataset.samples:
            target = sample['target']
            for i in range(len(target)):
                values[i].append(target[i])
        
        # Calculate statistics
        means = [np.mean(v) for v in values]
        mins = [np.min(v) for v in values]
        maxs = [np.max(v) for v in values]
        
        # Clear and redraw
        self.axes.clear()
        
        # Create bar positions
        x = np.arange(len(metrics))
        width = 0.35
        
        # Plot bars
        self.axes.bar(x, maxs, width, label='Max')
        self.axes.bar(x, mins, width, bottom=maxs, label='Min')
        
        # Add labels and legend
        self.axes.set_title('Expression Variety in Dataset')
        self.axes.set_xlabel('Facial Feature')
        self.axes.set_ylabel('Value Range')
        self.axes.set_xticks(x)
        self.axes.set_xticklabels(metrics)
        self.axes.legend()
        
        self.fig.tight_layout()
        self.draw()


class TrainingResultsPanel(QWidget):
    """Panel for visualizing and controlling the training process."""
    
    def __init__(self, session_dir=None, parent=None):
        """Initialize the training results panel.
        
        Args:
            session_dir: Directory containing the training session data
            parent: Parent widget
        """
        super().__init__(parent)
        self.session_dir = session_dir
        self.trainer = FacialFeatureTrainer()
        self.dataset = None
        self.training_thread = None
        
        # Setup UI
        self.setup_ui()
        
        # Load dataset if session directory is provided
        if session_dir:
            self.load_dataset(session_dir)
    
    def setup_ui(self):
        """Setup the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Create tabs for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Training tab
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        
        # Training parameters group
        params_group = QGroupBox("Training Parameters")
        params_layout = QHBoxLayout(params_group)
        
        # Left side parameters
        left_params = QVBoxLayout()
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("Epochs:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_spin)
        left_params.addLayout(epochs_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch Size:")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_spin)
        left_params.addLayout(batch_layout)
        
        # Right side parameters
        right_params = QVBoxLayout()
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_label = QLabel("Learning Rate:")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(self.lr_spin)
        right_params.addLayout(lr_layout)
        
        # Validation split
        val_layout = QHBoxLayout()
        val_label = QLabel("Validation Split:")
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.1, 0.5)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setDecimals(2)
        self.val_spin.setValue(0.2)
        val_layout.addWidget(val_label)
        val_layout.addWidget(self.val_spin)
        right_params.addLayout(val_layout)
        
        # Add parameter layouts to group
        params_layout.addLayout(left_params)
        params_layout.addLayout(right_params)
        training_layout.addWidget(params_group)
        
        # Training progress group
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bar and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Ready to train")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        # Training metrics plot
        self.metrics_plot = MetricsPlot(self, width=5, height=3)
        progress_layout.addWidget(self.metrics_plot)
        
        training_layout.addWidget(progress_group)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        
        buttons_layout.addWidget(self.train_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.save_button)
        training_layout.addLayout(buttons_layout)
        
        # Dataset tab
        dataset_tab = QWidget()
        dataset_layout = QVBoxLayout(dataset_tab)
        
        # Dataset info group
        dataset_info_group = QGroupBox("Dataset Information")
        dataset_info_layout = QVBoxLayout(dataset_info_group)
        
        # Dataset stats
        self.dataset_info_label = QLabel("No dataset loaded")
        dataset_info_layout.addWidget(self.dataset_info_label)
        
        # Expression variety plot
        self.expression_plot = ExpressionVarietyPlot(self, width=5, height=3)
        dataset_info_layout.addWidget(self.expression_plot)
        
        dataset_layout.addWidget(dataset_info_group)
        
        # Dataset quality group
        dataset_quality_group = QGroupBox("Dataset Quality")
        quality_layout = QVBoxLayout(dataset_quality_group)
        
        # Quality thresholds
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Quality Threshold:")
        self.quality_threshold = QDoubleSpinBox()
        self.quality_threshold.setRange(0.0, 1.0)
        self.quality_threshold.setSingleStep(0.05)
        self.quality_threshold.setValue(0.5)
        self.quality_threshold.setDecimals(2)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.quality_threshold)
        quality_layout.addLayout(threshold_layout)
        
        # Filter low quality samples checkbox
        self.filter_checkbox = QCheckBox("Filter Low Quality Samples")
        self.filter_checkbox.setChecked(True)
        quality_layout.addWidget(self.filter_checkbox)
        
        # Apply filters button
        self.apply_filters_button = QPushButton("Apply Filters")
        self.apply_filters_button.clicked.connect(self.apply_filters)
        quality_layout.addWidget(self.apply_filters_button)
        
        dataset_layout.addWidget(dataset_quality_group)
        
        # Add tabs
        self.tabs.addTab(training_tab, "Training")
        self.tabs.addTab(dataset_tab, "Dataset")
        
        # Results tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Model evaluation group
        eval_group = QGroupBox("Model Evaluation")
        eval_layout = QVBoxLayout(eval_group)
        
        # Evaluation metrics
        self.eval_metrics_label = QLabel("No evaluation results available")
        eval_layout.addWidget(self.eval_metrics_label)
        
        # Test on webcam button
        self.test_button = QPushButton("Test Model on Webcam")
        self.test_button.clicked.connect(self.test_model)
        self.test_button.setEnabled(False)
        eval_layout.addWidget(self.test_button)
        
        results_layout.addWidget(eval_group)
        self.tabs.addTab(results_tab, "Results")
    
    def load_dataset(self, session_dir):
        """Load dataset from the session directory.
        
        Args:
            session_dir: Directory containing the training session data
        """
        try:
            # Create dataset from session directory
            self.dataset = FacialFeatureDataset(session_dir)
            
            # Validate dataset loading
            if len(self.dataset.samples) == 0:
                raise ValueError("No valid samples found in the dataset")
            
            # Update UI with dataset information
            sample_count = len(self.dataset)
            sample = self.dataset.samples[0]
            
            self.dataset_info_label.setText(
                f"Dataset loaded from: {os.path.basename(session_dir)}\n"
                f"Total samples: {sample_count}\n"
                f"Features per sample: {sample['landmarks'].size}\n"
                f"Target metrics: 5 (eye ratios, mouth ratio, brow heights)"
            )
            
            # Update expression variety plot
            self.expression_plot.update_plot(self.dataset)
            
            # Enable training button if dataset is valid
            self.train_button.setEnabled(sample_count > 0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
            self.dataset = None
            self.dataset_info_label.setText("No dataset loaded")
            self.train_button.setEnabled(False)
    
    def apply_filters(self):
        """Apply quality filters to the dataset."""
        if not self.dataset or not hasattr(self.dataset, 'samples') or len(self.dataset.samples) == 0:
            return
            
        try:
            # Get quality threshold
            threshold = self.quality_threshold.value()
            
            # Filter samples if checkbox is checked
            if self.filter_checkbox.isChecked():
                original_count = len(self.dataset.samples)
                self.dataset.samples = [s for s in self.dataset.samples if s['quality'] >= threshold]
                filtered_count = len(self.dataset.samples)
                
                # Update dataset info
                self.dataset_info_label.setText(
                    f"Dataset: {os.path.basename(self.session_dir)}\n"
                    f"Original samples: {original_count}\n"
                    f"Filtered samples: {filtered_count}\n"
                    f"Removed {original_count - filtered_count} low quality samples"
                )
                
                # Update expression variety plot
                self.expression_plot.update_plot(self.dataset)
                
                QMessageBox.information(self, "Filters Applied", 
                    f"Applied quality threshold of {threshold:.2f}\n"
                    f"Removed {original_count - filtered_count} low quality samples")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply filters: {str(e)}")
    
    def start_training(self):
        """Start the model training process."""
        if not self.dataset or len(self.dataset) == 0:
            QMessageBox.warning(self, "Warning", "No dataset loaded or dataset is empty")
            return
            
        try:
            # Disable UI controls during training
            self.train_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(False)
            
            # Get training parameters
            params = {
                'epochs': self.epochs_spin.value(),
                'batch_size': self.batch_spin.value(),
                'val_split': self.val_spin.value(),
                'learning_rate': self.lr_spin.value()
            }
            
            # Create and start training thread
            self.training_thread = TrainingThread(self.trainer, self.dataset, params)
            
            # Connect signals
            self.training_thread.progress_updated.connect(self.update_progress)
            self.training_thread.epoch_completed.connect(self.update_metrics)
            self.training_thread.training_completed.connect(self.training_completed)
            self.training_thread.training_error.connect(self.training_error)
            
            # Start training
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training: {str(e)}")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def stop_training(self):
        """Stop the training process."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.status_label.setText("Training stopped by user")
    
    def update_progress(self, progress, status):
        """Update the progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
    
    def update_metrics(self, epoch, train_loss, val_loss):
        """Update the metrics plot with new epoch data.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
        """
        self.metrics_plot.update_plot(epoch, train_loss, val_loss)
    
    def training_completed(self, stats):
        """Handle training completion.
        
        Args:
            stats: Training statistics dictionary
        """
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)
        self.test_button.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"Training completed after {stats['epochs']} epochs")
        
        # Update evaluation metrics
        final_train_loss = stats['train_losses'][-1] if stats['train_losses'] else 0
        final_val_loss = stats['val_losses'][-1] if stats['val_losses'] else 0
        
        self.eval_metrics_label.setText(
            f"Training Results:\n"
            f"Final Training Loss: {final_train_loss:.6f}\n"
            f"Final Validation Loss: {final_val_loss:.6f}\n"
            f"Total Epochs: {stats['epochs']}\n"
            f"Model saved to: {os.path.join(self.trainer.model_dir, 'best_facial_model.pth')}"
        )
        
        # Update the metrics plot with all training data
        # This ensures the plot is created in the main UI thread
        if 'train_losses' in stats and 'val_losses' in stats:
            epochs = list(range(1, len(stats['train_losses']) + 1))
            for i, (train_loss, val_loss) in enumerate(zip(stats['train_losses'], stats['val_losses'])):
                self.metrics_plot.update_plot(epochs[i], train_loss, val_loss)
        
        # Show completion message
        QMessageBox.information(self, "Training Complete", 
            f"Model training completed successfully after {stats['epochs']} epochs.")
    
    def training_error(self, error_msg):
        """Handle training errors.
        
        Args:
            error_msg: Error message
        """
        self.train_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n{error_msg}")
    
    def save_model(self):
        """Save the trained model to a user-selected location."""
        if not self.trainer or not self.trainer.model:
            QMessageBox.warning(self, "Warning", "No trained model available to save")
            return
            
        try:
            # Get save path from user
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Model", 
                os.path.join(os.path.expanduser("~"), "facial_model.pth"),
                "PyTorch Model (*.pth)"
            )
            
            if save_path:
                # Save the model
                self.trainer.save_model(save_path)
                QMessageBox.information(self, "Model Saved", f"Model saved successfully to:\n{save_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
    
    def test_model(self):
        """Test the trained model on webcam input."""
        QMessageBox.information(self, "Test Model", 
            "This feature would open a webcam window to test the model in real-time.\n"
            "Implementation would connect to the webcam and apply the model to live facial landmarks.")
        # In a real implementation, this would launch a webcam window
        # and apply the trained model to live facial landmarks