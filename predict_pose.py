import csv
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),]
CSV_PATH = "hand_dataset.csv"
MODEL_PATH = "pose_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"
LEFT_MIN_PX = 10.0
LEFT_MAX_PX = 180.0
FACE_MODEL_PATH = "face_model.yml"
FACE_MATCH_THRESHOLD = 0.35
FACE_DISTANCE_SCALE = 100.0


def extract_features(joints):
    wrist = np.array(joints[0])
    features = []
    
    for joint in joints:
        relative = np.array(joint) - wrist
        features.extend(relative)
    
    finger_tips = [4, 8, 12, 16, 20]
    for tip_idx in finger_tips:
        dist = np.linalg.norm(np.array(joints[tip_idx]) - wrist)
        features.append(dist)
    
    palm_size = np.linalg.norm(np.array(joints[9]) - wrist)
    features.append(palm_size)
    
    if palm_size > 0:
        features = [f / palm_size for f in features]
    
    return features


def load_and_train_model():
    print("Loading")
    
    X = []
    y = []
    
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["handedness"] != "Right":
                continue
            
            joints = []
            valid = True
            for idx in range(21):
                x_val = row.get(f"joint{idx}_x", "")
                y_val = row.get(f"joint{idx}_y", "")
                if x_val == "" or y_val == "":
                    valid = False
                    break
                joints.append((float(x_val), float(y_val)))
            
            if valid and row["pose"] != "none":
                features = extract_features(joints)
                X.append(features)
                y.append(row["pose"])
    
    if len(X) == 0:
        print("No valid training data found!")
        return None, None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples with {len(np.unique(y))} classes: {np.unique(y)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("Training ANN model...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        learning_rate='adaptive',
        alpha=0.0001
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)
    
    print("Model saved successfully!")
    return model, scaler, label_encoder


def load_model():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
        print("Model files not found. Training new model...")
        return load_and_train_model()
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    
    print("Model loaded successfully!")
    return model, scaler, label_encoder


def draw_landmarks_on_image(
    image,
    detection_result,
    predicted_pose=None,
    confidence=None,
    left_distance=None,
    face_status=None,
):
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    
    for hand_landmarks in detection_result.hand_landmarks:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))
            cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)
        
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(annotated_image, points[start_idx], points[end_idx], (0, 255, 255), 2)
    
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

    if left_distance is not None:
        cv2.putText(
            annotated_image,
            f"Left thumb-index: {left_distance:.1f}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

    if face_status:
        cv2.putText(
            annotated_image,
            face_status,
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 200, 0),
            2,
        )
    
    return annotated_image


def predict_pose_realtime():
    model, scaler, label_encoder = load_model()
    
    if model is None:
        print("Failed to load or train model. Exiting...")
        return
    
    if not os.path.exists(FACE_MODEL_PATH):
        print(f"Error: {FACE_MODEL_PATH} not found!")
        print("Run enroll_face.py to record your face first.")
        return

    if not hasattr(cv2, "face"):
        print("Error: OpenCV face module not found.")
        print("Install opencv-contrib-python and try again.")
        return

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(FACE_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    print("\nStarting real-time prediction...")
    print("Press 'q' to quit, 'r' to retrain model")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(image)
        
        predicted_pose = None
        confidence = None
        left_distance = None
        face_ok = False
        face_status = "Face: not detected"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label, distance = face_recognizer.predict(face_roi)
            similarity = max(0.0, 1.0 - (distance / FACE_DISTANCE_SCALE))
            if similarity >= FACE_MATCH_THRESHOLD:
                face_ok = True
                face_status = f"Face: verified ({similarity:.0%})"
            else:
                face_status = f"Face: unknown ({similarity:.0%})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

        if face_ok and detection_result.hand_landmarks:
            height, width = frame.shape[:2]
            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = "unknown"
                if detection_result.handedness and detection_result.handedness[i]:
                    handedness = detection_result.handedness[i][0].category_name

                joints = []
                for idx in range(21):
                    lm = hand_landmarks[idx]
                    joints.append((int(lm.x * width), int(lm.y * height)))

                if handedness == "Right":
                    features = extract_features(joints)
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)
                    probabilities = model.predict_proba(features_scaled)[0]
                    confidence = probabilities.max()
                    predicted_pose = label_encoder.inverse_transform(prediction)[0]
                elif handedness == "Left":
                    distance_px = float(
                        np.linalg.norm(np.array(joints[4]) - np.array(joints[8]))
                    )
                    normalized = (distance_px - LEFT_MIN_PX) / (LEFT_MAX_PX - LEFT_MIN_PX)
                    left_distance = max(0.0, min(1.0, normalized)) * 100.0
        
        annotated = draw_landmarks_on_image(
            frame,
            detection_result,
            predicted_pose,
            confidence,
            left_distance,
            face_status,
        )
        cv2.imshow("Pose Prediction - Right Hand", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            print("\nRetraining model...")
            model, scaler, label_encoder = load_and_train_model()
            if model is None:
                print("Retraining failed. Exiting...")
                break
    
    cap.release()
    cv2.destroyAllWindows()


# Only run if executed directly, not when imported
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found!")
        print("Please run makedataset.py first to collect training data.")
    else:
        predict_pose_realtime()
