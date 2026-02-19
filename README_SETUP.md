# Vision Studio – Web UI + FastAPI

This project combines a React front-end with a FastAPI backend to provide a unified interface for face enrollment, hand pose dataset capture, and pose prediction.

## Project Structure

```
mp_project/
├── web-ui/                # React SPA (Create React App)
│   ├── src/
│   │   ├── App.js        # Multi-screen UI (Enroll / Dataset / Predict)
│   │   ├── App.css       # Styled components
│   │   └── index.js
│   ├── .env              # API base URL
│   ├── package.json
│   └── build/            # Built SPA (created after npm run build)
├── api.py                # FastAPI server + API endpoints
├── enroll_face.py        # Face enrollment logic
├── makedataset.py        # Hand pose dataset capture
├── predict_pose.py       # Pose prediction model
├── face_model.yml        # Trained face recognizer
├── hand_dataset.csv      # Hand pose training data
├── hand_landmarker.task  # MediaPipe hand landmarker model
└── README.md
```

## Setup

### 1. Install Python Dependencies

```bash
pip install fastapi uvicorn[standard] pillow opencv-contrib-python mediapipe numpy scikit-learn
```

### 2. Build the React App

```bash
cd web-ui
npm install
npm run build
cd ..
```

### 3. Start the FastAPI Server

```bash
python api.py
```

The server will:
- Serve the React app at `http://localhost:8000`
- Expose API endpoints at `http://localhost:8000/api/*`
- Auto-reload on Python file changes (development mode)

## API Endpoints

### POST `/api/enroll`
Enroll a face by name and image.

**Request:**
```
Content-Type: multipart/form-data
- name: str (form field)
- image: file (form file)
```

**Response:**
```json
{
  "message": "Face enrollment for 'John' completed successfully.",
  "ok": true
}
```

### POST `/api/dataset`
Initiate hand pose dataset capture.

**Request:**
```
Content-Type: multipart/form-data
- count: int (form field)
```

**Response:**
```json
{
  "message": "Dataset build initiated for 120 samples.",
  "ok": true
}
```

### POST `/api/predict`
Predict hand pose from an image.

**Request:**
```
Content-Type: multipart/form-data
- image: file (form file)
```

**Response:**
```json
{
  "message": "Pose prediction completed: open",
  "pose": "open",
  "ok": true
}
```

### GET `/api/health`
Health check.

**Response:**
```json
{
  "status": "ok"
}
```

## Development Workflow

### React Development (with Live Reload)

```bash
cd web-ui
npm start
```

This starts the local dev server at `http://localhost:3000` with hot reload. Update the `.env` file to point to your running FastAPI backend:

```
REACT_APP_API_BASE=http://localhost:8000
```

### FastAPI Development (with Auto-Reload)

```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Production Build

To build for production:

```bash
cd web-ui
npm run build
cd ..
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

The React build artifacts will be served as static files from the FastAPI server.

## Next Steps

1. **Integrate core logic:** Replace the TODO placeholders in `api.py` with actual calls to `enroll_face.py`, `makedataset.py`, and `predict_pose.py`.
2. **Error handling:** Add robust error handling and validation.
3. **CORS:** If running React and FastAPI on different ports during development, enable CORS in `api.py`:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```
4. **Testing:** Add unit and integration tests.
5. **Deployment:** Deploy to a cloud provider (Heroku, AWS, etc.) or self-host.
