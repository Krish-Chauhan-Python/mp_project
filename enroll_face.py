import os
import time

import cv2
import numpy as np


FACE_MODEL_PATH = "face_model.yml"
SAMPLES_TARGET = 60
CAPTURE_INTERVAL = 0.1


def main():
    if not hasattr(cv2, "face"):
        print("Error: OpenCV face module not found.")
        print("Install opencv-contrib-python and try again.")
        return

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    cap = cv2.VideoCapture(0)
    samples = []
    labels = []
    last_capture = 0.0

    print("Starting face enrollment...")
    print("Keep your face centered. Press 'q' to quit early.")

    while cap.isOpened() and len(samples) < SAMPLES_TARGET:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

            now = time.time()
            if now - last_capture >= CAPTURE_INTERVAL:
                face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                samples.append(face_roi)
                labels.append(0)
                last_capture = now

        cv2.putText(
            frame,
            f"Samples: {len(samples)}/{SAMPLES_TARGET}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Face Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < 5:
        print("Not enough samples captured. Try again.")
        return

    recognizer.train(samples, np.array(labels, dtype=np.int32))
    recognizer.save(FACE_MODEL_PATH)
    print(f"Saved face model to {FACE_MODEL_PATH}.")


if __name__ == "__main__":
    main()
