## Vision Studio - Hand Pose

Real-time hand pose recognition with MediaPipe and a web UI backed by a FastAPI
server. You can use the CLI tools for dataset capture
and live prediction, or run the web app for a guided workflow with gesture-to-action
mappings and live video processing.

## What Is Inside

- `api.py`: FastAPI server that serves the React UI and exposes REST endpoints.
- `web-ui/`: React single-page app (optional but recommended).
- `makedataset.py`: Collect hand landmark samples into `hand_dataset.csv`.
- `predict_pose.py`: Train (or load) a classifier and run real-time prediction.
- `enroll_face.py`: Capture a face model.
- `start.py`: Minimal hand landmark visualizer.

## Requirements

- Python 3.13+
- Webcam

Core dependencies are listed in `pyproject.toml`:

- mediapipe
- opencv-contrib-python
- numpy
- scikit-learn

If you want the API + web UI, install the extra packages:

- fastapi
- uvicorn[standard]
- pillow

## Setup

1. Create a virtual environment:

```bash
uv venv
```

2. Install dependencies:

```bash
uv pip install mediapipe opencv-contrib-python numpy scikit-learn
```

3. (Optional) Install API + web UI deps:

```bash
uv pip install fastapi uvicorn[standard] pillow
```

## Quick Start (CLI Workflow)

1. Enroll a face model:

```bash
uv run python enroll_face.py
```

This creates `face_model.yml`.

2. Collect training data for hand poses:

```bash
uv run python makedataset.py
```

- Use the buttons to set a pose label.
- Click Pause/Resume to start or stop recording.
- Samples are saved to `hand_dataset.csv`.

3. Train and run real-time prediction:

```bash
uv run python predict_pose.py
```

- The model will train if `pose_model.pkl` is missing.
- Press `q` to quit, `r` to retrain.

## Quick Start (Web UI + API)

1. Build the React app (one time or after UI changes):

```bash
cd web-ui
npm install
npm run build
cd ..
```

2. Start the API server (serves the UI and API):

```bash
uv run python api.py
```

Open `http://localhost:8000` in your browser.
Keep the API server running while using the frontend.

### Development Mode

Run React with hot reload:

```bash
cd web-ui
npm start
```

Set `REACT_APP_API_BASE` in `web-ui/.env` to point to the backend:

```
REACT_APP_API_BASE=http://localhost:8000
```

Run FastAPI with auto-reload:

```bash
uv run python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Gestures and Actions

- Gesture labels are stored in `gestures.json`.
- Gesture-to-action mappings are stored in `gesture_mappings.json`.
- Available actions include tab switching, media control, volume, and stop camera.

The API exposes endpoints to list and update gestures and mappings. See `api.py`
for details.

## Hand Poses

The default gesture labels are:

- up, down, left, right
- close, open
- yo, mf
- none

You can change or add labels in `gestures.json` or via the API.

## Files Generated

- `hand_dataset.csv`: Captured training data.
- `pose_model.pkl`: Trained classifier.
- `scaler.pkl`: Feature scaler.
- `encoder.pkl`: Label encoder.
- `face_model.yml`: Face model.

## Notes

- `hand_landmarker.task` must exist in the project root.
- Action execution and mouse control use Win32 APIs, so they are Windows-only.

## Troubleshooting

- If you see `OpenCV face module not found`, install `opencv-contrib-python`.
- If no training data is found, run `makedataset.py` and collect samples first.
- If the webcam does not open, check camera permissions or device index.

## License

Add a license file if you plan to share or distribute this project.
