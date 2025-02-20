
# Object Detection with SORT & DeepSORT

This project integrates object detection with **SORT** (Simple Online and Realtime Tracking) and **DeepSORT** (Deep Learning-based SORT) algorithms for real-time multi-object tracking. It uses object detection models to identify objects in each frame and then applies tracking to keep track of each object across frames.

## Features
- **Object Detection**: Detects objects using pre-trained models (YOLO)
- **Tracking**: Implements SORT and DeepSORT to track objects across multiple frames.
- **Real-time Processing**: Optimized for real-time video streams or recorded video files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmethamdiozen/object-tracking.git
   cd object-tracking
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained object detection model weights (e.g. YOLO) and place them in the appropriate directory.

## Usage

### Object Detection and Tracking
To run object detection with tracking using SORT algorithm, execute the following command:
```bash
python yolov8_sort.py
```

## Code Structure

- **detector.py**: Script that contains the object detection algorithm.
- **yolov8_sort.py**: Implements the SORT algorithm for object tracking.
- **yolov8_deepsort.py**: Implements the DeepSORT algorithm using appearance features for tracking.
- **sort.py**: The library that used in this project. To reach the original file https://github.com/abewley/sort
- **requirements.txt**: Python package dependencies.