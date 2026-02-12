import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from mediapipe import Image


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (1,5)
]


def draw_landmarks_on_image(image, detection_result):
    annotated_image = image.copy()
    height, width = annotated_image.shape[:2]
    for hand_landmarks in detection_result.hand_landmarks:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))
            cv2.circle(annotated_image, (x, y), 2, (255, 0, 0), -1)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(annotated_image, points[start_idx], points[end_idx], (0, 0, 255), 1)
    return annotated_image

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, image_bgr = cap.read()
    resized = cv2.resize(image_bgr, None ,fx=0.25, fy=0.25)
    if not ret:
        continue
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(resized, detection_result)
    cv2.imshow("MediaPipe Holistic", annotated_image)
    if cv2.waitKey(5) & 0xFF == ord(" "):
        break
cap.release()
cv2.destroyAllWindows()