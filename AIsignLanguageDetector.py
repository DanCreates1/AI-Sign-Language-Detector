import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import string
from sklearn.model_selection import train_test_split

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

letters = list(string.ascii_uppercase)

data = []
labels_list = []

current_letter = None
run_detection = False
test_acc_percent = None

print("Press A-Z to collect samples for that letter.")
print("Press Q to stop and train the model.")
print("Press R to skip training and start detection immediately.")

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    landmarks = []
    wrist_x, wrist_y = hand.landmark[0].x, hand.landmark[0].y
    for lm in hand.landmark:
        landmarks.append(lm.x - wrist_x)
        landmarks.append(lm.y - wrist_y)
    return landmarks

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Current letter: {current_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Samples collected: {len(data)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("ASL Trainer", frame)

    key = cv2.waitKey(1) & 0xFF

    if 65 <= key <= 90 or 97 <= key <= 122:
        current_letter = chr(key).upper()
        print(f"Collecting samples for: {current_letter}")

    if key == ord('q'):
        break

    if key == ord('r'):
        run_detection = True
        break

    if current_letter:
        landmarks = extract_landmarks(results)
        if landmarks:
            data.append(landmarks)
            labels_list.append(current_letter)

cap.release()
cv2.destroyAllWindows()

if not run_detection:
    X = np.array(data)
    y = np.array([letters.index(l) for l in labels_list])
    y = tf.keras.utils.to_categorical(y, num_classes=26)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    test_acc_percent = test_acc * 100
    print(f"Final Test Accuracy: {test_acc_percent:.2f}%")

    model.save("asl_model.h5")
    print("Model trained and saved!")
else:
    print("Skipping training. Accuracy on screen will show N/A unless trained in this run.")

model = tf.keras.models.load_model("asl_model.h5")
cap = cv2.VideoCapture(0)
print("Starting real-time ASL detection. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "None"
    confidence = 0.0

    landmarks = extract_landmarks(results)
    if landmarks:
        input_data = np.array(landmarks).reshape(1, -1)
        probs = model.predict(input_data, verbose=0)
        class_id = np.argmax(probs)
        prediction = letters[class_id]
        confidence = float(np.max(probs)) * 100

    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    acc_text = f"{test_acc_percent:.2f}%" if test_acc_percent is not None else "N/A"

    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(frame, f"Test Accuracy: {acc_text}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ASL Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()