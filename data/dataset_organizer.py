import os
import random
import shutil
import logging
from pathlib import Path
from typing import Dict
from config.config_settings import DOWNLOAD_DIR, PROCESSED_DIR, DATASET_SPLIT

logger = logging.getLogger(__name__)

# Fixed order of categories (regardless of which categories are available in the dataset)
FIXED_CATEGORIES = [
    "feldsalat", "weeds", "beetroot", "coriander", "lettuce", "rucola",
    "strawberry", "chilli", "wildsalat", "onion"
]

class DatasetOrganizer:
    """Organizer for dataset preprocessing and organization."""
    
    def __init__(self):
        """Initialize dataset organizer."""
        self.source_path = DOWNLOAD_DIR
        self.output_path = PROCESSED_DIR
        self.split_ratios = DATASET_SPLIT
    
    def create_directories(self):
        """Create target directories for dataset splits."""
        for split in self.split_ratios.keys():
            (self.output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_path / split / "labels").mkdir(parents=True, exist_ok=True)
    
    def move_files(self, files: list, category: str, split: str):
        """
        Move image and label files to respective split directories.
        
        Args:
            files: List of image filenames
            category: Category name (e.g., 'strawberry')
            split: Split name (e.g., 'train')
        """
        category_path = self.source_path / category
        label_path = category_path / "labels"
        
        for file in files:
            # Copy image
            shutil.copy(
                category_path / file, 
                self.output_path / split / "images" / file
            )
            
            # Copy label if exists
            label_file = label_path / file.replace(".jpg", ".txt")
            if label_file.exists():
                shutil.copy(
                    label_file, 
                    self.output_path / split / "labels" / file.replace(".jpg", ".txt")
                )
    
    def get_split_counts(self, total: int) -> Dict[str, int]:
        """
        Calculate number of files for each split.
        
        Args:
            total: Total number of files
            
        Returns:
            Dict with count for each split
        """
        return {
            'train': int(total * self.split_ratios['train']),
            'val': int(total * self.split_ratios['val']),
            'test': int(total * self.split_ratios['test'])
        }
    
    def organize_dataset(self):
        """
        Organize dataset into train/val/test splits.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.create_directories()
            
            # Get class folders (feldsalat, strawberry, etc.)
            categories = [d for d in os.listdir(self.source_path) 
                         if (self.source_path / d).is_dir()]
            
            if not categories:
                logger.warning("No category directories found in source path")
                return False
            
            # Process each category
            for category in categories:
                category_path = self.source_path / category
                
                # Skip if not a directory or if 'labels' is the only directory
                if not category_path.is_dir() or category == 'labels':
                    continue
                
                # Get image filenames
                image_files = [f for f in os.listdir(category_path) if f.endswith(".jpg")]
                if not image_files:
                    logger.warning(f"No images found in category: {category}")
                    continue
                
                # Shuffle files for random split
                random.shuffle(image_files)
                
                # Calculate split sizes
                counts = self.get_split_counts(len(image_files))
                
                # Split dataset
                start_idx = 0
                split_files = {}
                
                for split, count in counts.items():
                    end_idx = start_idx + count
                    split_files[split] = image_files[start_idx:end_idx]
                    start_idx = end_idx
                
                # Add any remaining files to test split
                if start_idx < len(image_files):
                    split_files['test'].extend(image_files[start_idx:])
                
                # Move files to respective folders
                for split, files in split_files.items():
                    self.move_files(files, category, split)
            
            logger.info("Dataset successfully reorganized into train/val/test!")
            
            # Create data.yaml file
            self.create_data_yaml(categories)
            
            return True
        except Exception as e:
            logger.error(f"Failed to organize dataset: {e}")
            return False
    
    def create_data_yaml(self, categories):
        """
        Create data.yaml file for YOLO training.
        
        Args:
            categories: List of category names
        """
        # Ensure that categories are in the fixed order
        categories_in_order = [category for category in FIXED_CATEGORIES if category in categories]
        
        # Number of categories that are actually present in the dataset
        # Keep the fixed number of categories (10)
        nc = 10  # Always 10, regardless of dataset size
        
        yaml_path = self.output_path / "data.yaml"
        
        with open(yaml_path, 'w') as f:
            f.write(f"# YOLOv8 dataset configuration\n")
            f.write(f"path: {self.output_path}\n")
            f.write(f"train: train/images\n")
            f.write(f"val: val/images\n")
            f.write(f"test: test/images\n\n")
            f.write(f"nc: {nc}\n")  # Always 10 categories
            f.write(f"names: {FIXED_CATEGORIES}\n")  # List all categories in fixed order
        
        logger.info(f"Created data.yaml file at {yaml_path}")
