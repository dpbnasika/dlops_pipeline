"""Configuration settings for the YOLO Pipeline project."""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data_storage"
DOWNLOAD_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# YOLO model settings
YOLO_CONFIG = {
    "task": "detect",
    "model": "yolov8s.pt",
    "epochs": 30,
    "batch_size": 16,
    "img_size": 640,
    "device": 0,
}

# Dataset split ratios
DATASET_SPLIT = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

# Firebase settings
FIREBASE_BUCKET = "dataset-collection-c967a.appspot.com"
FIREBASE_PREFIX = "yolo_1/"

# Monitoring settings
MONITOR_INTERVAL = 60  # seconds
MIN_NEW_FILES_THRESHOLD = 1  # minimum number of new files to trigger pipeline

# Export settings
EXPORT_FORMAT = "torchscript"

# Create required directories
def create_directories():
    """Create necessary directories if they don't exist."""
    for split in DATASET_SPLIT:
        (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    (DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "exported_models").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "predictions").mkdir(parents=True, exist_ok=True)
