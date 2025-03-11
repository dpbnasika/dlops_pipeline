"""Main entry point for the YOLO Pipeline project."""

import argparse
import logging
from pathlib import Path
from config.config_settings import create_directories
from pipeline.pipeline_manager import PipelineManager
from utils.file_utils import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Pipeline")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring Firebase for new data")
    parser.add_argument("--run", action="store_true", help="Run the pipeline once")
    parser.add_argument("--test-image", type=str, help="Path to test image for prediction")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    """Main function to run the YOLO Pipeline."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    # Create necessary directories
    create_directories()
    
    # Initialize pipeline manager
    pipeline_manager = PipelineManager()
    
    # Process test image path if provided
    test_image = Path(args.test_image) if args.test_image else None
    
    if args.run:
        # Run pipeline once
        pipeline_manager.run_pipeline(test_image)
    elif args.monitor:
        # Start monitoring
        pipeline_manager.monitor()
    else:
        print("No action specified. Use --monitor or --run")
        print("For help, use --help")

if __name__ == "__main__":
    main()
