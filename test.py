import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from src.utils.hands_utils import get_both_hands
from src.utils.finger_utils import finger_open
from src.utils.finger_utils import get_center
from src.utils.finger_utils import distance

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model_path = "models/landmark_model.h5"
model = load_model(model_path)

video = cv2.VideoCapture(0)

classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

last_pred = None
word = ""
gesture_count = 0
prev_x = None
movement_threshold = 100
last_time = 0
cooldown = 1.5
buffer = ""
space_count = 0
SPACE_FRAME = 5
SPACE_DISPLAY_DURATION = 1.0
space_added_time = 0


def delet_word():
    global word, buffer
    if len(word) > 0:
        buffer = word[-1] + buffer
        word = word[:-1]


def undo_word():
    global word, buffer
    if len(buffer) > 0:
        word += buffer[0]
        buffer = buffer[1:]


def add_space():
    global word
    if not word.endswith(" "):
        word += " "
        print("Space added:", repr(word))


with mp_hands.Hands(max_num_hands=2, model_complexity=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while True:
        ret, img = video.read()
        frame = cv2.flip(img, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                dist = distance(thumb_tip,index_tip)

                if dist < 0.04:
                    if finger_open(hand_landmarks) == 3:
                        space_count+=1
                        if(space_count == SPACE_FRAME):
                            add_space()
                            space_added_time = time.time()
                            space_count = 0
                else:
                    space_count = 0
                if time.time() - space_added_time < SPACE_DISPLAY_DURATION:
                    cv2.putText(frame, "SPACE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
                
                
                hand = handedness.classification[0].label
                cen_x, cen_y = get_center(hand_landmarks, w, h)
                if hand == 'Left':
                    if finger_open(hand_landmarks) >= 4:
                        if prev_x is not None:
                            if (cen_x - prev_x) > movement_threshold and (time.time() - last_time > cooldown):
                                delet_word()
                                last_time = time.time()
                if hand == 'Right':
                    if finger_open(hand_landmarks) >= 4:
                        if prev_x is not None:
                            if (prev_x - cen_x) > movement_threshold and (time.time() - last_time > cooldown):
                                undo_word()
                                last_time = time.time()
                prev_x = cen_x
           

            x_coords = [(lm.x * w) for landmarks in results.multi_hand_landmarks for lm in landmarks.landmark]
            y_coords = [(lm.y * h) for landmarks in results.multi_hand_landmarks for lm in landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            row = get_both_hands(results, frame)
            row_np = np.array(row).reshape(1, -1)
            pred_prob = model.predict(row_np)
            pred_idx = np.argmax(pred_prob)
            prediction = classes[pred_idx]
            confidence = pred_prob[0][pred_idx]

            cv2.rectangle(frame, (x_min - 20, y_min - 60), (x_min + 120, y_min - 20), (255, 0, 255), -1)
            display_text = f'{prediction} {confidence*100:.1f}%'
            cv2.putText(frame, display_text, (x_min, y_min - 35),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 255), 5)

            if prediction == last_pred:
                print(f'gesture {last_pred} {gesture_count}')
                gesture_count += 1
            else:
                last_pred = prediction

            if gesture_count == 20:
                print(f'{last_pred} printed')
                gesture_count = 0
                word += last_pred

            h, w, _ = frame.shape
            x_coords = [(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
        else:
            gesture_count = 0
            last_pred = None

        cv2.putText(frame, word, (100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (5, 25, 55), 2)
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

cv2.destroyAllWindows()
