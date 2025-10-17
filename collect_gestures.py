import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# List of gestures
gestures = ["open_palm", "index_up", "thumbs_up", "peace"]
num_samples = 200  # Number of frames per gesture
data_path = "gesture_data"

# Create gesture_data folder if it doesn't exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    for label, gesture in enumerate(gestures):
        print(f"Collecting data for gesture: {gesture}")
        data = []
        time.sleep(2)  # small delay before starting

        collected = 0
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    flattened = [coord for point in landmarks for coord in point]
                    flattened.append(label)  # append gesture label
                    data.append(flattened)
                    collected += 1

            # Display instructions on screen
            cv2.putText(frame, f"Gesture: {gesture} | Samples: {collected}/{num_samples}",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Collecting Gestures", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save collected data for this gesture
        np.save(os.path.join(data_path, f"{gesture}.npy"), np.array(data))
        print(f"Saved {collected} samples for {gesture}")

cap.release()
cv2.destroyAllWindows()
print("Data collection complete!")
