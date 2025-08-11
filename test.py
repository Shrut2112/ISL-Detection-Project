import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
from src.utils.hands_utils import get_both_hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model_path = "models/landmark_model.h5"
model = load_model(model_path)

video = cv2.VideoCapture(0)

classes  = ['1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

last_pred = None
word = ""
gesture_count = 0

with mp_hands.Hands(max_num_hands = 2,model_complexity =1, min_detection_confidence = 0.7,min_tracking_confidence = 0.7) as hands:

    while True:

        ret,img = video.read()
        frame = cv2.flip(img,1)
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)


        if results.multi_hand_landmarks:
            # passing the result of the frame to detect if there is multiple hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            h,w,_ = frame.shape
            x_coords = [(lm.x*w) for landmarks in results.multi_hand_landmarks for lm in landmarks.landmark]
            y_coords = [(lm.y*h) for landmarks in results.multi_hand_landmarks for lm in landmarks.landmark]
            x_min,x_max = int(min(x_coords)),int(max(x_coords))
            y_min,y_max = int(min(y_coords)),int(max(y_coords))


            row = get_both_hands(results,frame)
            row_np = np.array(row).reshape(1,-1)

            pred_idx = model.predict(row_np)
            prediction = classes[np.argmax(pred_idx)]

            cv2.rectangle(frame,(x_min-20,y_min-60),(x_min+30,y_min-20),(255,0,255),-1)
            cv2.putText(frame,prediction,(x_min,y_min-30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),4)
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (255,0,255), 5)

            if prediction == last_pred:
                print(f'gesture {last_pred} {gesture_count}')
                gesture_count+=1
            else:
                last_pred = prediction
                
            if gesture_count == 20:
                print(f'{last_pred} printed')
                gesture_count = 0
                word += last_pred           

            h,w,_ = frame.shape
            x_coords = [(lm.x*w) for lm in hand_landmarks.landmark]
            y_coords = [(lm.y*h) for lm in hand_landmarks.landmark]

            x_min,x_max = int(min(x_coords)),int(max(x_coords))
            y_min,y_max = int(min(y_coords)),int(max(y_coords))       
        else:
            gesture_count = 0
            last_pred = None

        
        
        cv2.putText(frame,word,(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(5,25,55),2)
    
        cv2.imshow('MediaPipe Hands',frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
cv2.destroyAllWindows()

