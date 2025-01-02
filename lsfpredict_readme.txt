# YOLOv7 Hand Detection with MediaPipe

This project combines YOLOv7 for object detection and MediaPipe for hand landmark detection. The script `yolomedia.py` captures video from a webcam, detects hands using YOLOv7, and then uses MediaPipe to segment and draw landmarks on the detected hands.

## Setup

### Prerequisites

Ensure you have Python 3.6 or later installed. You will also need to install the required Python packages.

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/yolov7-hand-detection.git
    cd yolov7-hand-detection
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the YOLOv7 weights:
    - Download the `yolov7-tiny.pt` weights from the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7).
    - Place the weights file in the `src/yolov7` directory.

4. Ensure the `best.pt` weights file (your trained weights) is also placed in the `src/yolov7` directory.

### Running the Script

To run the hand detection and segmentation script, execute the following command:
```sh
python yolomedia.py
```

### Usage

- The script will start capturing video from your webcam.
- It will detect hands using YOLOv7 and segment them using MediaPipe.
- Press `q` to quit the application.

### Sign Language Prediction

The `lsfpredict.py` script extends the functionality of `yolomedia.py` by adding sign language prediction.

### Additional Setup for `lsfpredict.py`

1. Ensure you have the following additional files in the project directory:
    - `scaler_1.pkl`
    - `scaler_2.pkl`
    - `best_model_1.pkl`
    - `best_model_2.pkl`
    - `best_model_3.pkl`

2. These files are required for the sign language prediction model.

### Running the Sign Language Prediction Script

To run the sign language detection and prediction script, execute the following command:
```sh
python lsfpredict.py
```

### Usage

- The script will start capturing video from your webcam.
- It will detect hands using YOLOv7, extract landmarks using MediaPipe, and predict sign language gestures.
- Press `q` to quit the application.

### Notes

- You can adjust the confidence thresholds and other parameters in the `detect_and_segment_hands` function as needed.
- Ensure your webcam is properly connected and accessible.

## Acknowledgements

- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [MediaPipe](https://mediapipe.dev/)
