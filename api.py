"""
FastAPI server for face enrollment, hand dataset capture, and pose prediction.
Serves the React UI from web-ui/build and exposes REST API endpoints.
"""

import os
import base64
import json
import time
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import Image as MPImage
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage

# Import prediction logic
from predict_pose import (
    extract_features,
    load_model,
    load_and_train_model,
    HAND_CONNECTIONS,
    FACE_MATCH_THRESHOLD,
    FACE_DISTANCE_SCALE,
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
)

try:
    import ctypes
except Exception:
    ctypes = None

app = FastAPI()

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROJECT_ROOT = Path(__file__).parent
WEB_UI_BUILD = PROJECT_ROOT / "web-ui" / "build"
FACE_MODEL_PATH = PROJECT_ROOT / "face_model.yml"
HAND_DATASET_PATH = PROJECT_ROOT / "hand_dataset.csv"
HAND_LANDMARKER_PATH = PROJECT_ROOT / "hand_landmarker.task"
GESTURES_PATH = PROJECT_ROOT / "gestures.json"
MAPPINGS_PATH = PROJECT_ROOT / "gesture_mappings.json"

# Global state for models
_model_cache = {"model": None, "scaler": None, "encoder": None}
_detector_cache = {"detector": None}
_last_action = {"pose": None, "time": 0.0}
_stop_camera_hold = {"start_time": None, "gesture": None}
_screen_cache = {"width": None, "height": None}
_last_mouse_update = {"time": 0.0}
_left_click_state = {"is_held": False}

ACTION_COOLDOWN = 1.0
MOUSE_SENSITIVITY = 1.3  # Increase this for more sensitive mouse movement (1.0 = normal, 1.5 = 50% more sensitive)
ACTIONS = {
    "none": "No action",
    "next_tab": "Next tab",
    "prev_tab": "Previous tab",
    "switch_window": "Switch window",
    "play_pause": "Play or pause",
    "next_track": "Next track",
    "prev_track": "Previous track",
    "volume_up": "Volume up",
    "volume_down": "Volume down",
    "stop_camera": "Stop camera",
}


def _get_model():
    """Lazily load and cache the model."""
    if _model_cache["model"] is None:
        print("Loading model on first request...")
        model, scaler, encoder = load_model()
        _model_cache["model"] = model
        _model_cache["scaler"] = scaler
        _model_cache["encoder"] = encoder
    return _model_cache["model"], _model_cache["scaler"], _model_cache["encoder"]


def _load_json(path, default):
    if not path.exists():
        path.write_text(json.dumps(default, indent=2))
        return default
    return json.loads(path.read_text() or "{}")


def _save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def _get_gestures():
    default = {
        "gestures": ["up", "down", "left", "right", "close", "open", "yo", "mf"]
    }
    data = _load_json(GESTURES_PATH, default)
    return data.get("gestures", [])


def _get_mappings():
    default = {"mappings": {gesture: "none" for gesture in _get_gestures()}}
    data = _load_json(MAPPINGS_PATH, default)
    if "mappings" not in data:
        data["mappings"] = {}
    for gesture in _get_gestures():
        data["mappings"].setdefault(gesture, "none")
    _save_json(MAPPINGS_PATH, data)
    return data["mappings"]


def _save_mapping(gesture, action):
    data = _load_json(MAPPINGS_PATH, {"mappings": {}})
    data.setdefault("mappings", {})
    data["mappings"][gesture] = action
    _save_json(MAPPINGS_PATH, data)


def _ensure_dataset_csv():
    if not HAND_DATASET_PATH.exists():
        header = ["timestamp", "pose", "handedness"]
        for idx in range(21):
            header.append(f"joint{idx}_x")
            header.append(f"joint{idx}_y")
        HAND_DATASET_PATH.write_text(",".join(header) + "\n")


def _append_dataset_row(pose, handedness, points):
    row = [str(time.time()), pose, handedness]
    for point in points:
        if point is None:
            row.extend(["", ""])
        else:
            row.extend([str(point[0]), str(point[1])])
    with HAND_DATASET_PATH.open("a", encoding="utf-8") as handle:
        handle.write(",".join(row) + "\n")


def _count_pose_samples(pose):
    if not HAND_DATASET_PATH.exists():
        return 0
    count = 0
    with HAND_DATASET_PATH.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) > 1 and parts[1] == pose:
                count += 1
    return count


def _clear_pose_samples(pose):
    if not HAND_DATASET_PATH.exists():
        return 0

    removed = 0
    lines = []
    with HAND_DATASET_PATH.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        lines.append(header)
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) > 1 and parts[1] == pose:
                removed += 1
            else:
                lines.append(line)

    with HAND_DATASET_PATH.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)

    return removed


def _press_vk(vk_code):
    """Press and release a single virtual key."""
    if ctypes is None:
        return False
    try:
        ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)  # Key down
        ctypes.windll.user32.keybd_event(vk_code, 0, 2, 0)  # Key up
        return True
    except Exception:
        return False


def _move_mouse_win32(x, y):
    """Move mouse cursor using Win32 API SetCursorPos."""
    if ctypes is None:
        return False
    try:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        return True
    except Exception as e:
        print(f"Mouse move error: {e}")
        return False


def _get_screen_size():
    """Get screen dimensions using Win32 API GetSystemMetrics."""
    if ctypes is None:
        return 1920, 1080
    try:
        SM_CXSCREEN = 0  # Screen width
        SM_CYSCREEN = 1  # Screen height
        width = ctypes.windll.user32.GetSystemMetrics(SM_CXSCREEN)
        height = ctypes.windll.user32.GetSystemMetrics(SM_CYSCREEN)
        return width, height
    except Exception:
        return 1920, 1080


def _mouse_left_down():
    """Simulate left mouse button down."""
    if ctypes is None:
        return False
    try:
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
        return True
    except Exception:
        return False


def _mouse_left_up():
    """Simulate left mouse button up."""
    if ctypes is None:
        return False
    try:
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
        return True
    except Exception:
        return False


def _detect_closed_hand(hand_landmarks):
    """Detect if hand is closed (fist) by checking distance from wrist to fingers."""
    if len(hand_landmarks) < 21:
        return False
    
    # Wrist is landmark 0
    wrist = hand_landmarks[0]
    wrist_pos = (float(wrist.x), float(wrist.y))
    
    # Check distances from wrist to fingertips
    # Fingertips: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    fingertip_indices = [4, 8, 12, 16, 20]
    distances = []
    
    for idx in fingertip_indices:
        tip = hand_landmarks[idx]
        tip_pos = (float(tip.x), float(tip.y))
        # Euclidean distance
        dist = ((tip_pos[0] - wrist_pos[0]) ** 2 + (tip_pos[1] - wrist_pos[1]) ** 2) ** 0.5
        distances.append(dist)
    
    # If average distance is small, hand is closed
    avg_distance = sum(distances) / len(distances)
    threshold = 0.15  # Adjust this to tune detection sensitivity
    
    return avg_distance < threshold


def _press_hotkey_win32(modifier_vks, key_vk):
    """Press a hotkey combination using Win32 API (more reliable than pyautogui)."""
    if ctypes is None:
        return False
    try:
        # Press all modifiers
        for mod_vk in modifier_vks:
            ctypes.windll.user32.keybd_event(mod_vk, 0, 0, 0)
        
        # Press and release the main key
        ctypes.windll.user32.keybd_event(key_vk, 0, 0, 0)
        ctypes.windll.user32.keybd_event(key_vk, 0, 2, 0)
        
        # Release modifiers in reverse order
        for mod_vk in reversed(modifier_vks):
            ctypes.windll.user32.keybd_event(mod_vk, 0, 2, 0)
        
        return True
    except Exception:
        return False


def _execute_action(action):
    if not action or action == "none":
        return False

    # VK codes for Windows
    VK_CONTROL = 0x11
    VK_MENU = 0x12  # Alt key
    VK_SHIFT = 0x10
    VK_TAB = 0x09

    # Media keys via Win32 for reliability
    if action == "play_pause":
        return _press_vk(0xB3)
    if action == "next_track":
        return _press_vk(0xB0)
    if action == "prev_track":
        return _press_vk(0xB1)
    if action == "volume_up":
        # Press volume up 5 times to shift by 10 (2 per press)
        for _ in range(5):
            _press_vk(0xAF)
        return True
    if action == "volume_down":
        # Press volume down 5 times to shift by 10 (2 per press)
        for _ in range(5):
            _press_vk(0xAE)
        return True

    # Keyboard shortcuts via Win32 for reliability
    if action == "next_tab":
        return _press_hotkey_win32([VK_CONTROL], VK_TAB)
    elif action == "prev_tab":
        return _press_hotkey_win32([VK_CONTROL, VK_SHIFT], VK_TAB)
    elif action == "switch_window":
        return _press_hotkey_win32([VK_MENU], VK_TAB)
    elif action == "stop_camera":
        # Stop camera signal - handled by frontend
        return True
    
    return False


# ============================================================================
# Helper Functions
# ============================================================================

def _load_detector():
    """Load MediaPipe hand detector."""
    if _detector_cache["detector"] is None:
        if not HAND_LANDMARKER_PATH.exists():
            raise FileNotFoundError(f"Hand landmarker model not found: {HAND_LANDMARKER_PATH}")
        base_options = python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        _detector_cache["detector"] = vision.HandLandmarker.create_from_options(options)
    return _detector_cache["detector"]


def _draw_landmarks_on_image(image, detection_result, predicted_pose=None, confidence=None):
    """Draw hand landmarks and pose prediction on the image."""
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    # Draw hand landmarks
    for hand_landmarks in detection_result.hand_landmarks:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))
            cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(annotated_image, points[start_idx], points[end_idx], (0, 255, 255), 2)
    
    # Draw predicted pose
    if predicted_pose:
        text = f"Pose: {predicted_pose}"
        if confidence is not None:
            text += f" ({confidence:.1%})"
        cv2.putText(
            annotated_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    
    return annotated_image


# ============================================================================
# API Routes
# ============================================================================

@app.post("/api/predict-frame")
async def predict_frame(frame_data: str = Form(...)):
    """
    Predict pose from a single video frame (base64 encoded).
    
    Request:
        - frame_data: str (form field) - Base64-encoded JPEG frame
    
    Response:
        - annotated_frame: str - Base64-encoded annotated frame
        - pose: str - Predicted pose label
        - confidence: float - Confidence of prediction
        - ok: bool
    """
    try:
        # Decode base64 frame
        try:
            frame_bytes = base64.b64decode(frame_data.split(',')[-1])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Failed to decode frame", "ok": False},
                )
        except Exception as decode_error:
            print(f"Frame decode error: {decode_error}")
            return JSONResponse(
                status_code=400,
                content={"message": f"Frame decode error: {str(decode_error)}", "ok": False},
            )
        
        # Load model and detector
        try:
            model, scaler, label_encoder = _get_model()
            print(f"Model loaded: {model is not None}, Scaler: {scaler is not None}, Encoder: {label_encoder is not None}")
        except Exception as model_error:
            print(f"Model load error: {model_error}")
            return JSONResponse(
                status_code=400,
                content={"message": f"Model load error: {str(model_error)}", "ok": False},
            )
        
        if model is None:
            return JSONResponse(
                status_code=400,
                content={"message": "Model not available", "ok": False},
            )
        
        try:
            detector = _load_detector()
            print(f"Detector loaded: {detector is not None}")
        except Exception as detector_error:
            print(f"Detector load error: {detector_error}")
            return JSONResponse(
                status_code=400,
                content={"message": f"Detector load error: {str(detector_error)}", "ok": False},
            )
        
        # Detect hand landmarks
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = detector.detect(mp_image)
            print(f"Hands detected: {len(detection_result.hand_landmarks)}")
        except Exception as detect_error:
            print(f"Detection error: {detect_error}")
            return JSONResponse(
                status_code=400,
                content={"message": f"Detection error: {str(detect_error)}", "ok": False},
            )
        
        predicted_pose = None
        confidence = None
        action = None
        action_executed = False
        mouse_pos = None
        
        # Detect hands
        height, width = frame.shape[:2]
        right_hand_found = False
        left_hand_found = False
        
        # Predict pose from right hand AND track left hand for mouse
        if detection_result.hand_landmarks:
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = "unknown"
                if detection_result.handedness and detection_result.handedness[i]:
                    handedness = detection_result.handedness[i][0].category_name
                
                print(f"Hand {i}: {handedness}")
                
                if handedness == "Right" and not right_hand_found:
                    right_hand_found = True
                    joints = []
                    for idx in range(21):
                        lm = hand_landmarks[idx]
                        joints.append((int(lm.x * width), int(lm.y * height)))
                    
                    features = extract_features(joints)
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = float(probabilities.max())
                    
                    # Only accept prediction if confidence >= 90%
                    if confidence >= 0.9:
                        predicted_pose = str(label_encoder.inverse_transform(prediction)[0])
                        print(f"Prediction: {predicted_pose} ({confidence:.1%})")
                    else:
                        print(f"Prediction below 90% threshold: {confidence:.1%} - ignoring")
                
                elif handedness == "Left" and not left_hand_found:
                    left_hand_found = True
                    print("Left hand detected - starting mouse control")
                    
                    # Use wrist (landmark 0) to control mouse
                    wrist = hand_landmarks[0]
                    
                    # Normalize coordinates (0-1)
                    norm_x = float(wrist.x)
                    norm_y = float(wrist.y)
                    
                    # Mirror X coordinate because camera is typically flipped
                    norm_x = 1.0 - norm_x
                    
                    # Get screen dimensions (cached)
                    if _screen_cache["width"] is None or _screen_cache["height"] is None:
                        screen_width, screen_height = _get_screen_size()
                        _screen_cache["width"] = screen_width
                        _screen_cache["height"] = screen_height
                    
                    screen_width = _screen_cache["width"]
                    screen_height = _screen_cache["height"]
                    
                    # Apply sensitivity scaling around the center
                    # This amplifies small hand movements for better screen coverage
                    scaled_x = (norm_x - 0.5) * MOUSE_SENSITIVITY + 0.5
                    scaled_y = (norm_y - 0.5) * MOUSE_SENSITIVITY + 0.5
                    
                    # Clamp to screen bounds (0-1 range)
                    scaled_x = max(0, min(1, scaled_x))
                    scaled_y = max(0, min(1, scaled_y))
                    
                    # Convert normalized coordinates to screen coordinates
                    # X: 0 = left, 1 = right; Y: 0 = top, 1 = bottom
                    screen_x = int(scaled_x * screen_width)
                    screen_y = int(scaled_y * screen_height)
                    
                    # Move mouse using Win32 API (more reliable than pyautogui)
                    # Update every 2 frames to balance responsiveness and performance
                    now = time.time()
                    if now - _last_mouse_update["time"] >= 0.016:  # ~60fps
                        if _move_mouse_win32(screen_x, screen_y):
                            mouse_pos = {"x": screen_x, "y": screen_y}
                            _last_mouse_update["time"] = now
                            print(f"Mouse -> ({screen_x}, {screen_y})")
                    
                    # Check if hand is closed for left click
                    hand_is_closed = _detect_closed_hand(hand_landmarks)
                    
                    if hand_is_closed and not _left_click_state["is_held"]:
                        # Hand just closed - press left mouse button
                        _mouse_left_down()
                        _left_click_state["is_held"] = True
                        print("Left click DOWN")
                    elif not hand_is_closed and _left_click_state["is_held"]:
                        # Hand opened - release left mouse button
                        _mouse_left_up()
                        _left_click_state["is_held"] = False
                        print("Left click UP")
        
        # If no left hand detected, release mouse button if held
        if not left_hand_found and _left_click_state["is_held"]:
            _mouse_left_up()
            _left_click_state["is_held"] = False
            print("Left click released (hand lost)")

        stop_camera = False
        
        if predicted_pose:
            mappings = _get_mappings()
            action = mappings.get(predicted_pose, "none")
            now = time.time()
            
            # Handle stop_camera with 3-second continuous hold requirement
            if action == "stop_camera":
                # Check if same gesture is being held
                if _stop_camera_hold["gesture"] == predicted_pose and _stop_camera_hold["start_time"] is not None:
                    # Same gesture - check if held for 3+ seconds
                    hold_duration = now - _stop_camera_hold["start_time"]
                    if hold_duration >= 3.0:
                        action_executed = True
                        stop_camera = True
                        # Reset hold state after triggering
                        _stop_camera_hold["gesture"] = None
                        _stop_camera_hold["start_time"] = None
                        print(f"Stop camera triggered after {hold_duration:.1f}s hold")
                else:
                    # New/different gesture - start hold timer
                    _stop_camera_hold["gesture"] = predicted_pose
                    _stop_camera_hold["start_time"] = now
                    print(f"Stop camera hold started for {predicted_pose}")
            else:
                # Not stop_camera - reset hold state
                if _stop_camera_hold["gesture"] is not None:
                    _stop_camera_hold["gesture"] = None
                    _stop_camera_hold["start_time"] = None
                
                # Normal action handling with cooldown
                if (
                    action
                    and action != "none"
                    and (predicted_pose != _last_action["pose"]
                         or (now - _last_action["time"]) > ACTION_COOLDOWN)
                ):
                    action_executed = _execute_action(action)
                    if action_executed:
                        _last_action["pose"] = predicted_pose
                        _last_action["time"] = now
        else:
            # No pose detected - reset hold state and set action to "no_action"
            if _stop_camera_hold["gesture"] is not None:
                _stop_camera_hold["gesture"] = None
                _stop_camera_hold["start_time"] = None
            action = "no_action"
        
        print(f"Final result: pose={predicted_pose}, conf={confidence}, action={action}")
        
        # Draw landmarks on frame
        annotated_frame = _draw_landmarks_on_image(
            frame, detection_result, predicted_pose, confidence
        )
        
        # Encode annotated frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate hold duration if stop_camera is active
        hold_duration = None
        if action == "stop_camera" and _stop_camera_hold["start_time"] is not None:
            hold_duration = time.time() - _stop_camera_hold["start_time"]
        
        return {
            "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}",
            "pose": predicted_pose,
            "confidence": confidence,
            "action": action,
            "action_executed": action_executed,
            "stop_camera": stop_camera,
            "hold_duration": hold_duration,
            "mouse_pos": mouse_pos,
            "ok": True
        }
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"message": f"Prediction failed: {str(e)}", "ok": False},
        )


@app.post("/api/dataset")
async def build_dataset(count: int = Form(...)):
    """
    Build a hand pose dataset by capturing N samples.
    
    Request:
        - count: int (form field) - Number of samples to capture
    
    Response:
        - message: str - Status or result
    """
    try:
        # TODO: Call your makedataset.py main function here
        message = f"Dataset build initiated for {count} samples."
        return {"message": message, "ok": True}

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Dataset build failed: {str(e)}", "ok": False},
        )


@app.post("/api/dataset/capture-frame")
async def capture_dataset_frame(
    gesture: str = Form(...), frame_data: str = Form(...)
):
    """
    Capture a single labeled frame for dataset creation.

    Request:
        - gesture: str (form field) - Gesture label
        - frame_data: str (form field) - Base64-encoded JPEG frame
    """
    try:
        if gesture not in _get_gestures():
            return JSONResponse(
                status_code=400,
                content={"message": "Unknown gesture label", "ok": False},
            )

        frame_bytes = base64.b64decode(frame_data.split(",")[-1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(
                status_code=400,
                content={"message": "Failed to decode frame", "ok": False},
            )

        detector = _load_detector()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        captured = False
        reason = "no_hand_detected"

        if detection_result.hand_landmarks:
            height, width = frame.shape[:2]
            selected = None
            selected_handedness = "unknown"

            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = "unknown"
                if detection_result.handedness and detection_result.handedness[i]:
                    handedness = detection_result.handedness[i][0].category_name

                if handedness == "Right":
                    selected = hand_landmarks
                    selected_handedness = handedness
                    break

                if selected is None:
                    selected = hand_landmarks
                    selected_handedness = handedness

            if selected is not None:
                points = []
                for idx in range(21):
                    lm = selected[idx]
                    points.append((int(lm.x * width), int(lm.y * height)))

                _ensure_dataset_csv()
                _append_dataset_row(gesture, selected_handedness, points)
                captured = True
                reason = "captured"

        total_for_pose = _count_pose_samples(gesture)
        return {
            "ok": True,
            "captured": captured,
            "pose": gesture,
            "total_for_pose": total_for_pose,
            "reason": reason,
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Capture failed: {str(e)}", "ok": False},
        )


@app.delete("/api/dataset/{gesture_name}")
async def delete_dataset_samples(gesture_name: str):
    pose = gesture_name.strip().lower()
    if pose not in _get_gestures():
        return JSONResponse(
            status_code=400,
            content={"message": "Unknown gesture label", "ok": False},
        )

    removed = _clear_pose_samples(pose)
    total_for_pose = _count_pose_samples(pose)
    return {
        "ok": True,
        "pose": pose,
        "removed": removed,
        "total_for_pose": total_for_pose,
    }


@app.post("/api/predict")
async def predict_pose(image: UploadFile = File(...)):
    """
    Predict hand pose from an image.
    
    Request:
        - image: UploadFile (form file) - Image for pose inference
    
    Response:
        - message: str - Status or result
        - pose: str - Predicted pose label (optional)
    """
    try:
        # Read the uploaded image
        contents = await image.read()
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Convert to OpenCV format
        pil_img = PILImage.open(tmp_path)
        img_array = np.array(pil_img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        predicted_pose = "unknown"
        message = f"Pose prediction completed: {predicted_pose}"

        # Clean up
        os.unlink(tmp_path)

        return {"message": message, "pose": predicted_pose, "ok": True}

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Prediction failed: {str(e)}", "ok": False},
        )


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/actions")
async def list_actions():
    return {"actions": ACTIONS}


@app.get("/api/gestures")
async def list_gestures():
    return {"gestures": _get_gestures()}


@app.post("/api/gestures")
async def add_gesture(payload: dict = Body(...)):
    name = str(payload.get("name", "")).strip().lower()
    if not name:
        return JSONResponse(status_code=400, content={"message": "Name required"})

    gestures = _get_gestures()
    if name in gestures:
        return {"ok": True, "gestures": gestures}

    gestures.append(name)
    _save_json(GESTURES_PATH, {"gestures": gestures})
    _save_mapping(name, "none")
    return {"ok": True, "gestures": gestures}


@app.delete("/api/gestures/{gesture_name}")
async def delete_gesture(gesture_name: str):
    name = gesture_name.strip().lower()
    gestures = _get_gestures()
    if name not in gestures:
        return {"ok": True, "gestures": gestures}

    gestures = [g for g in gestures if g != name]
    _save_json(GESTURES_PATH, {"gestures": gestures})

    data = _load_json(MAPPINGS_PATH, {"mappings": {}})
    data.setdefault("mappings", {})
    if name in data["mappings"]:
        data["mappings"].pop(name)
        _save_json(MAPPINGS_PATH, data)

    return {"ok": True, "gestures": gestures}


@app.get("/api/mappings")
async def list_mappings():
    return {"mappings": _get_mappings()}


@app.put("/api/mappings")
async def update_mapping(payload: dict = Body(...)):
    gesture = str(payload.get("gesture", "")).strip().lower()
    action = str(payload.get("action", "none")).strip().lower()

    if gesture not in _get_gestures():
        return JSONResponse(status_code=400, content={"message": "Unknown gesture"})
    if action not in ACTIONS:
        return JSONResponse(status_code=400, content={"message": "Unknown action"})

    _save_mapping(gesture, action)
    return {"ok": True, "mappings": _get_mappings()}


@app.post("/api/retrain")
async def retrain_model():
    try:
        model, scaler, encoder = load_and_train_model()
        _model_cache["model"] = model
        _model_cache["scaler"] = scaler
        _model_cache["encoder"] = encoder
        return {"ok": True, "message": "Retraining complete"}
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": f"Retraining failed: {str(exc)}"},
        )


# ============================================================================
# Serve React SPA
# ============================================================================

if WEB_UI_BUILD.exists():
    @app.get("/")
    async def serve_root():
        return FileResponse(WEB_UI_BUILD / "index.html")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api"):
            return JSONResponse(status_code=404, content={"detail": "Not Found"})

        candidate = WEB_UI_BUILD / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(WEB_UI_BUILD / "index.html")
else:
    @app.get("/")
    async def root():
        return {
            "message": "React app not built yet.",
            "instructions": [
                "1. cd web-ui",
                "2. npm install (if needed)",
                "3. npm run build",
                "4. Restart the FastAPI server",
            ],
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
