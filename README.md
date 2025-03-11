# A time-efficient YOLO pipeline project for object detection models

A full-scale, object-oriented pipeline for data preprocessing, training, predicting, and exporting YOLO models with Firebase integration.

## Features

- Google Firebase integration for dataset retrieval
- Automatic dataset organization and splitting
- YOLO model training with configurable parameters
- Model inference and export functionality
- Continuous monitoring of Firebase for new data
- Pipeline automation based on new data threshold

## Project Structure

```
yolo_pipeline/
├── config/             # Configuration settings
├── data/               # Data management (Firebase, dataset)
├── models/             # YOLO model operations
├── utils/              # Utility functions
├── pipeline/           # Pipeline orchestration
├── tests/              # Unit tests
├── main.py             # Entry point
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Firebase credentials:
   - open `.env` file 
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your Firebase credentials JSON file

4. Source the environment variables:
   ```bash
   source .env   
   ```
5. Run it using various arguments:
   ```bash
   python3 main.py --monitor --debug --run --test-image=test_images/strawberry.png
   ```
6. The results will be present in the data_storage folder under predict and you could use the .torchscript with the cpp script and libtorch library for faster inferencing.