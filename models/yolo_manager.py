"""YOLO model manager for training, prediction, and export operations."""

import os
import subprocess
import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from config.config_settings import YOLO_CONFIG, PROCESSED_DIR, DATA_DIR

logger = logging.getLogger(__name__)

class YOLOManager:
    """Manager for YOLO model operations."""
    
    def __init__(self):
        """Initialize YOLO model manager."""
        self.model_name = YOLO_CONFIG["model"]
        self.task = YOLO_CONFIG["task"]
        self.epochs = YOLO_CONFIG["epochs"]
        self.batch_size = YOLO_CONFIG["batch_size"]
        self.img_size = YOLO_CONFIG["img_size"]
        
        # Check for GPU availability
        if torch.cuda.is_available():
            self.device = YOLO_CONFIG["device"]
        else:
            logger.warning("CUDA not available, using CPU instead.")
            self.device = "cpu"  # Fallback to CPU if no GPU is detected
        
        # Paths
        self.data_yaml = PROCESSED_DIR / "data.yaml"
        self.runs_dir = DATA_DIR / "runs"
        self.export_dir = DATA_DIR / "exported_models"
        
        # Create directories
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def run_command(self, command: str) -> bool:
        """
        Execute a command as a subprocess.
        
        Args:
            command: Command to run
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Running command: {command}")
            subprocess.run(command, shell=True, check=True)
            logger.info(f"Command successful: {command}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return False
    
    def train(self) -> bool:
        """
        Train the YOLO model.
        
        Returns:
            bool: True if training successful, False otherwise
        """
        if not self.data_yaml.exists():
            logger.error(f"Data YAML file not found: {self.data_yaml}")
            return False
        
        train_command = (
            f"yolo task={self.task} mode=train "
            f"model={self.model_name} "
            f"data={self.data_yaml} "
            f"epochs={self.epochs} batch={self.batch_size} imgsz={self.img_size} device={self.device} "
            f"project={self.runs_dir} name=train exist_ok=True"
        )
        
        logger.info("Starting model training...")
        return self.run_command(train_command)
    
    def predict(self, image_path: Union[str, Path]) -> bool:
        """
        Run prediction on an image using the trained model.
        
        Args:
            image_path: Path to the image for prediction
            
        Returns:
            bool: True if prediction successful, False otherwise
        """
        model_path = self.runs_dir / "train" / "weights" / "best.pt"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            return False
        
        predict_command = (
            f"yolo task={self.task} mode=predict "
            f"model={model_path} "
            f"source={image_path} "
            f"project={DATA_DIR / 'predictions'} name=predict exist_ok=True"
        )
        
        logger.info(f"Starting prediction on image {image_path}...")
        return self.run_command(predict_command)
    
    def export(self, format: str = "torchscript") -> Optional[Path]:
        """
        Export the trained model to the specified format.
        
        Args:
            format: Export format (default: torchscript)
            
        Returns:
            Optional[Path]: Path to the exported model if successful, None otherwise
        """
        model_path = self.runs_dir / "train" / "weights" / "best.pt"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.export_dir / f"yolov8_best_{timestamp}.{format}"
        
        export_command = (
            f"yolo export model={model_path} format={format} imgsz={self.img_size}"
        )
        
        logger.info(f"Exporting model to {format} format...")
        success = self.run_command(export_command)
        
        if not success:
            return None
        
        # Dynamically find the exported model path
        exported_model_dir = self.runs_dir / "train" / "weights"
        exported_model_files = list(exported_model_dir.glob(f"*.{format}"))
        
        if not exported_model_files:
            logger.error(f"No exported model found in {exported_model_dir}")
            return None
        
        # Take the most recent exported file
        exported_model_path = max(exported_model_files, key=os.path.getctime)
        
        # Move the exported model to the desired directory with a timestamped name
        os.rename(exported_model_path, export_path)
        logger.info(f"Model saved to {export_path}")
        
        return export_path
