import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import string
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ==== SETUP ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

letters = list(string.ascii_uppercase)  # A–Z
data, labels = [], []
sample_counts = defaultdict(int)
collecting = True

print("📸 ASL Data Collection + Training")
print("Press A–Z to collect samples")
print("Press SPACE to train model")
print("Press R to run detection")
print("Press Q to quit")

cap = cv2.VideoCapture(0)
current_label = None
model = None

# ==== NORMALIZED LANDMARK EXTRACTION ====
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    wrist_x = hand.landmark[0].x
    wrist_y = hand.landmark[0].y

    landmarks = []
    for lm in hand.landmark:
        landmarks.append(lm.x - wrist_x)
        landmarks.append(lm.y - wrist_y)

    return landmarks

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw first hand
    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1) & 0xFF

    # ==== PRIORITY KEYS FIRST ====
    if key == ord(' '):  # Train model
        if len(data) < 50:
            print("❌ Need more samples before training!")
            continue

        print("⚙️ Training model...")
        X = np.array(data)
        y = np.array([letters.index(label) for label in labels])
        y = tf.keras.utils.to_categorical(y, num_classes=26)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(42,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(26, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

        model.save("asl_model_landmarks.h5")
        print("✅ Model Saved! Press R to test live.")
        collecting = False

    elif key == ord('r'):  # Run detection 
        if model is None:
            try:
                model = tf.keras.models.load_model("asl_model_landmarks.h5")
            except:
                print("❌ No trained model found! Train first.")
                continue

        print("🎥 LIVE MODE (Press Q to stop)")
        while True:
            success, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                landmarks = extract_landmarks(results)
                if landmarks:
                    pred = model.predict(np.expand_dims(landmarks, 0), verbose=0)
                    letter = letters[np.argmax(pred)]
                    cv2.putText(frame, letter, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

            cv2.imshow("ASL Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("🛑 Live mode stopped.")

    elif key == ord('q'):  # Quit
        break

    # ==== LETTER SAMPLE COLLECTION ====
    elif key in [ord(c.lower()) for c in letters] or key in [ord(c.upper()) for c in letters]:
        current_label = chr(key).upper()
        print(f"🖐 Collecting samples for: {current_label}")

    if current_label and results.multi_hand_landmarks:
        landmarks = extract_landmarks(results)
        if landmarks:
            data.append(landmarks)
            labels.append(current_label)
            sample_counts[current_label] += 1

    # ==== ON-SCREEN INFO ====
    cv2.putText(frame, f"Collect: {current_label if current_label else '-'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Samples: {sample_counts[current_label] if current_label else 0}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Total: {len(data)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Trainer", frame)

cap.release()
cv2.destroyAllWindows()
print("👋 Program Ended")
