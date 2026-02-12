import csv , os , time
import tkinter as tk
import cv2
import mediapipe as mp
from mediapipe import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),(0, 5), (5, 6), (6, 7), (7, 8),(5, 9), (9, 10), (10, 11), (11, 12),(9, 13), (13, 14), (14, 15), (15, 16),(13, 17), (17, 18), (18, 19), (19, 20),(0, 17), (0, 9), (0, 13)]

POSES = ["up", "down", "left", "right", "close", "open", "yo" , "mf" , "none"]
CAPTURE_INTERVAL = 0.05

def draw_landmarks_on_image(image, detection_result):
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    for hand_landmarks in detection_result.hand_landmarks:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))
            cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(annotated_image, points[start_idx], points[end_idx], (0, 255, 255), 1)
    return annotated_image


def build_header():
    header = ["timestamp", "pose", "handedness"]
    for idx in range(21):
        header.append(f"joint{idx}_x")
        header.append(f"joint{idx}_y")
    return header


def ensure_csv():
    with open("hand_dataset.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(build_header())


def append_row(pose, handedness, points):
    row = [time.time(), pose, handedness]
    for point in points:
        if point is None:
            row.extend(["", ""])
        else:
            row.extend([point[0], point[1]])
    with open("hand_dataset.csv", "a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


cap = cv2.VideoCapture(0)
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

ensure_csv()

current_pose = "none"
paused = True
last_capture = 0.0
running = True


def set_pose(pose):
    global current_pose
    current_pose = pose
    pose_label.config(text=f"Pose: {current_pose}")


def toggle_pause():
    global paused
    paused = not paused
    pause_label.config(text=f"Paused: {paused}")


def on_close():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


def update_frame():
    global last_capture
    if not running:
        return
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(image)

    annotated = draw_landmarks_on_image(frame, detection_result)
    cv2.imshow("Hand Capture", annotated)
    cv2.waitKey(1)

    now = time.time()
    if not paused and (now - last_capture) >= CAPTURE_INTERVAL:
        last_capture = now
        height, width = frame.shape[:2]
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            handedness = "unknown"
            if detection_result.handedness and detection_result.handedness[0]:
                handedness = detection_result.handedness[0][0].category_name
            
            if handedness == "Right":
                points = []
                for idx in range(21):
                    lm = hand_landmarks[idx]
                    points.append((int(lm.x * width), int(lm.y * height)))
                append_row(current_pose, handedness, points)

    root.after(10, update_frame)


root = tk.Tk()
root.title("Hand Data Collector")

pose_label = tk.Label(root, text=f"Pose: {current_pose}")
pose_label.pack(padx=10, pady=5)

pause_label = tk.Label(root, text=f"Paused: {paused}")
pause_label.pack(padx=10, pady=5)

button_frame = tk.Frame(root)
button_frame.pack(padx=10, pady=5)

for pose in POSES:
    button = tk.Button(button_frame, text=pose, width=8, command=lambda p=pose: set_pose(p))
    button.pack(side=tk.LEFT, padx=3, pady=3)

pause_button = tk.Button(root, text="Pause/Resume", width=12, command=toggle_pause)
pause_button.pack(padx=10, pady=10)

root.protocol("WM_DELETE_WINDOW", on_close)
root.after(10, update_frame)
root.mainloop()