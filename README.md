## MediaPipe Hand Pose + Face Gate

Real-time hand pose recognition using MediaPipe, with a simple face verification gate.
It includes tools to collect hand pose data, train a small classifier, and run a
live prediction demo from your webcam.

## What Is Inside

- `makedataset.py`: Collect hand landmark samples into `hand_dataset.csv`.
- `predict_pose.py`: Train (or load) a classifier and run real-time prediction.
- `enroll_face.py`: Capture a face model for verification.
- `start.py`: Minimal hand landmark visualizer.

## Requirements

- Python 3.13+
- Webcam

Dependencies are listed in `pyproject.toml`:

- mediapipe
- opencv-contrib-python
- numpy
- scikit-learn

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install mediapipe opencv-contrib-python numpy scikit-learn
```

## Quick Start

1. Enroll a face model (required for prediction):

```bash
python enroll_face.py
```

This creates `face_model.yml`.

2. Collect training data for hand poses:

```bash
python makedataset.py
```

- Use the buttons to set a pose label.
- Click Pause/Resume to start or stop recording.
- Samples are saved to `hand_dataset.csv`.

3. Train and run real-time prediction:

```bash
python predict_pose.py
```

- The model will train if `pose_model.pkl` is missing.
- Press `q` to quit, `r` to retrain.

## Hand Poses

The capture app includes these labels by default:

- up, down, left, right
- close, open
- yo, mf
- none

You can change or add labels in `makedataset.py` by editing `POSES`.

## Files Generated

- `hand_dataset.csv`: Captured training data.
- `pose_model.pkl`: Trained classifier.
- `scaler.pkl`: Feature scaler.
- `encoder.pkl`: Label encoder.
- `face_model.yml`: Face recognition model.

## Notes

- `hand_landmarker.task` must exist in the project root.
- Face verification uses OpenCV's contrib face module. Make sure you installed
	`opencv-contrib-python`, not just `opencv-python`.
- Prediction only runs when a face is verified.

## Troubleshooting

- If you see `OpenCV face module not found`, install `opencv-contrib-python`.
- If no training data is found, run `makedataset.py` and collect samples first.
- If the webcam does not open, check camera permissions or device index.

## License

Add a license file if you plan to share or distribute this project.
