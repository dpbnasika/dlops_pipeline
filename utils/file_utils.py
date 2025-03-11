"""File utility functions for the YOLO Pipeline project."""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def get_file_count(file_path: Path) -> int:
    """
    Get file count from a file.
    
    Args:
        file_path: Path to the file containing the count
        
    Returns:
        int: File count
    """
    try:
        with open(file_path, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def write_file_count(file_path: Path, count: int):
    """
    Write file count to a file.
    
    Args:
        file_path: Path to write count
        count: Count to write
    """
    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(str(count))
    logger.debug(f"Updated file count to {count} in {file_path}")

def find_test_image() -> Optional[Path]:
    """
    Find a test image in the dataset.
    
    Returns:
        Optional[Path]: Path to a test image, None if not found
    """
    from config.config_settings import PROCESSED_DIR
    
    test_dir = PROCESSED_DIR / "test" / "images"
    
    if test_dir.exists():
        image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            return test_dir / image_files[0]
    
    logger.warning("No test images found")
    return None

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )
