import os
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Wczytanie modelu
model = load_model("model/emotion_model_v6.keras")

# Klasy emocji
classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Detektor twarzy (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Kamera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48))

        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        face_normalized = face_gray / 255.0

        face_input = np.expand_dims(face_normalized, axis=(0, -1))  # -> (1, 48, 48, 1)

        prediction = model.predict(face_input)
        emotion = classes[np.argmax(prediction)]
        propability = np.max(prediction)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion}: {propability}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
