"""Pipeline manager for orchestrating the entire pipeline."""

import time
import logging
from pathlib import Path
from typing import Optional
from data.firebase_manager import FirebaseManager
from data.dataset_organizer import DatasetOrganizer
from models.yolo_manager import YOLOManager
from utils.file_utils import get_file_count, write_file_count, find_test_image
from config.config_settings import DATA_DIR, MIN_NEW_FILES_THRESHOLD, MONITOR_INTERVAL

logger = logging.getLogger(__name__)

class PipelineManager:
    """Manager for orchestrating the entire pipeline."""
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.firebase_manager = FirebaseManager()
        self.dataset_organizer = DatasetOrganizer()
        self.yolo_manager = YOLOManager()
        self.count_file = DATA_DIR / "last_file_count.txt"
    
    def run_pipeline(self, test_image: Optional[Path] = None) -> bool:
        """
        Run the complete pipeline: fetch, organize, train, predict, export.
        
        Args:
            test_image: Path to test image for prediction (optional)
            
        Returns:
            bool: True if pipeline successful, False otherwise
        """
        logger.info("Starting pipeline execution")
        
        # Step 1: Fetch new data from Firebase
        logger.info("Step 1: Fetching data from Firebase")
        new_files_count, _ = self.firebase_manager.download_new_files()

        print(new_files_count)
        
        if new_files_count == 0:
            logger.info("No new files to process, pipeline completed")
            #return True
        
        # Step 2: Organize dataset
        logger.info("Step 2: Organizing dataset")
        if not self.dataset_organizer.organize_dataset():
            logger.error("Failed to organize dataset, pipeline aborted")
            return False
        
        # Step 3: Train model
        logger.info("Step 3: Training model")
        if not self.yolo_manager.train():
            logger.error("Failed to train model, pipeline aborted")
            return False
        
        # Step 4: Predict on test image if provided, otherwise find one
        logger.info("Step 4: Running prediction")
        if test_image is None:
            test_image = find_test_image()
        
        if test_image and test_image.exists():
            if not self.yolo_manager.predict(test_image):
                logger.warning("Prediction failed, continuing pipeline")
        else:
            logger.warning("No test image available for prediction, skipping step")
        
        # Step 5: Export model
        logger.info("Step 5: Exporting model")
        exported_model = self.yolo_manager.export()
        
        if exported_model is None:
            logger.error("Failed to export model, pipeline aborted")
            return False
        
        logger.info(f"Pipeline completed successfully. Model exported to {exported_model}")
        return True
    
    def monitor(self):
        """
        Continuously monitor Firebase for new files and trigger the pipeline when necessary.
        """
        logger.info("Starting Firebase monitoring...")
        
        while True:
            try:
                existing_files = self.firebase_manager.get_existing_files()
                firebase_files = self.firebase_manager.get_firebase_files()
                
                # Calculate number of new files
                new_files = firebase_files - existing_files
                new_file_count = len(new_files)
                
                logger.info(f"Checking for new files... Found {new_file_count} new files.")
                
                # Get the last known file count
                last_count = get_file_count(self.count_file)
                
                # Trigger pipeline if threshold reached
                if new_file_count >= MIN_NEW_FILES_THRESHOLD:
                    logger.info(f"New files threshold reached. Triggering pipeline...")
                    success = self.run_pipeline()
                    
                    if success:
                        # Update file count after processing
                        write_file_count(self.count_file, len(firebase_files))
                
                # Wait before checking again
                time.sleep(MONITOR_INTERVAL)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Wait before trying again
                time.sleep(MONITOR_INTERVAL)
