import cv2
import mediapipe as mp
import joblib
from collections import deque, Counter
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load the trained model
model = joblib.load("gesture_model.pkl")
gestures = ["open_palm", "index_up", "thumbs_up", "peace"]  # gesture list

# Rolling window for smoothing predictions
window_size = 5
predictions_window = deque(maxlen=window_size)

# Cooldown dictionary to avoid repeated triggers
cooldown_time = 0.5  # seconds
last_triggered = {gesture: 0 for gesture in gestures}

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        gesture_name = ""

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                flattened = [coord for point in landmarks for coord in point]

                # Predict gesture
                prediction = model.predict([flattened])[0]
                predictions_window.append(prediction)

                # Most common prediction in the rolling window
                most_common_pred = Counter(predictions_window).most_common(1)[0][0]
                gesture_name = gestures[int(most_common_pred)]

                # --- DEBUG PRINT ---
                print("Predicted label:", prediction, "Gesture:", gesture_name)

                # Media control actions with cooldown
                current_time = time.time()
                if current_time - last_triggered[gesture_name] > cooldown_time:
                    if gesture_name == "open_palm":
                        print("Action: Play/Pause Music")
                        pyautogui.press("space")  # Play/Pause
                    elif gesture_name == "index_up":
                        print("Action: Next Track")
                        pyautogui.hotkey("ctrl", "right")  # Next Track with Ctrl+Right
                    elif gesture_name == "thumbs_up":
                        print("Action: Volume Up")
                        pyautogui.press("volumeup")  # Volume Up
                    elif gesture_name == "peace":
                        print("Action: Volume Down")
                        pyautogui.press("volumedown")  # Volume Down
                    last_triggered[gesture_name] = current_time

                # Display gesture name on screen
                cv2.putText(frame, f"{gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show output
        cv2.imshow("Hand Tracking", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
