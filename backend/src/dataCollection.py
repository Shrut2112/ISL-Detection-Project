import cv2
import mediapipe as mp
import numpy as np
import copy
import csv
import itertools
from hands_utils import get_both_hands

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

video = cv2.VideoCapture(0)

classes  = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

current_index = 0
current_label = classes[current_index]
count_for_label = 0
filepath = 'keyPoints.csv'

def write_in_csv(path,row):
    #appending in csv file
    with open(path,'a',newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

with mp_hands.Hands(max_num_hands = 2,model_complexity =1, min_detection_confidence = 0.7,min_tracking_confidence = 0.7) as hands:

    while True:

        ret,frame = video.read()

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
                
            if cv2.waitKey(1) & 0xFF == ord('c'):
                if count_for_label < 100:
                    count_for_label+=1
                    row = get_both_hands(results,current_label,frame)
                    write_in_csv(filepath,row)
                    print(f'{current_label} written at row {count_for_label}')
                else:
                    image_cap = 0
                    current_index += 1
                    count_for_label = 0
                    if(current_index > len(classes)-1):
                        break
                    current_label = classes[current_index]
        
    # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands',cv2.flip(frame,1))
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
cv2.destroyAllWindows()

