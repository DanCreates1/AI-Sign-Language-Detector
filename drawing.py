import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Canvas for drawing
canvas = None

# Webcam
cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
drawing = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            h, w, c = img.shape

            # Index fingertip (8)
            x1 = int(hand_landmarks.landmark[8].x * w)
            y1 = int(hand_landmarks.landmark[8].y * h)

            # Thumb tip (4)
            x2 = int(hand_landmarks.landmark[4].x * w)
            y2 = int(hand_landmarks.landmark[4].y * h)

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Distance between index and thumb
            distance = math.hypot(x2 - x1, y2 - y1)

            # If fingers are touching (pinch)
            if distance < 40:
                drawing = True
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            else:
                drawing = False
                prev_x, prev_y = 0, 0

            # Draw on canvas
            if drawing:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1

                cv2.line(canvas, (prev_x, prev_y), (x1, y1), (255, 0, 255), 5)
                prev_x, prev_y = x1, y1

    # Merge canvas and webcam
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Air Drawing", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()