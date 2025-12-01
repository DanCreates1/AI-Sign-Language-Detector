import cv2 
import mediapipe as mp
import numpy as np
import tensorflow as tf
import string
from sklearn.model_selection import train_test_split


#makes a hand object
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=3,
    min_detection_confidence=0.5,
)

mp_draw = mp.solutions.drawing_utils

#captures video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


landmarks = []

if results.multi_hand_landmarks:
    hand = results.multi_hand_landmarks[0]

    wrist_x = hand.landmark[0].x
    wrist_y = hand.landmark[0].y

    for lm in hand.landmark:
        landmarks.append(lm.x - wrist_x)
        landmarks.append(lm.y - wrist_y)    


data = []
lables = []
current_letter = 'A'  # Example

if landmarks:
    data.append(landmarks)
    labels.append(current_letter)


letters = list(string.ascii_uppercase)

y = np.array([letters.index(l) for l in letters])

y = tf.keras.utils.to_categorical(y, num_classes=26)

# Convert data to numpy array
X = np.array(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cap.release()
cv2.destroyAllWindows()