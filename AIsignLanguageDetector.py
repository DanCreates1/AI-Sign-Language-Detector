import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import string

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

letters = list(string.ascii_uppercase)

data = []     
labels_list = []  

current_letter = None
run_detection = False

#print instructions
print("Press A–Z to collect samples for that letter.")
print("Press Q to stop and train the model.")
print("Press R to skip training and start detection immediately.")

# Function to extract normalized landmarks
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

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show info
    cv2.putText(frame, f"Current letter: {current_letter}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Samples collected: {len(data)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("ASL Trainer", frame)

    key = cv2.waitKey(1) & 0xFF

    
    if key in [ord(c) for c in letters]:
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

# Train model if not skipping training
if not run_detection:
    X = np.array(data)
    y = np.array([letters.index(l) for l in labels_list])
    y = tf.keras.utils.to_categorical(y, num_classes=26)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build model layers 
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
    model.save("asl_model.h5")
    print("Model trained and saved!")


model = tf.keras.models.load_model("asl_model.h5")
cap = cv2.VideoCapture(0)
print("Starting real-time ASL detection. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "None"
    landmarks = extract_landmarks(results)
    if landmarks:
        input_data = np.array(landmarks).reshape(1, -1)
        probs = model.predict(input_data, verbose=0)
        class_id = np.argmax(probs)
        prediction = letters[class_id]

    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    # Show prediction
    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ASL Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
