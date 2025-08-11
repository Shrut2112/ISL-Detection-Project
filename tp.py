import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


video = cv2.VideoCapture(0)
sentence = "Hello Shrut"

def finger_open(hand_landmarks):
     tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pink
     open = 0
     for tip in tips[1:]:
          if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
               open+=1
     return open >= 4
def get_center(hand_landmarks,w,h):
     return int(hand_landmarks.landmark[9].x*w),int(hand_landmarks.landmark[9].y*h)


prev_x = None
movement_threshold = 100
last_time = 0
cooldown = 1.5
with mp_hands.Hands(max_num_hands = 2,model_complexity =1, min_detection_confidence = 0.7,min_tracking_confidence = 0.7) as hands:

    while True:

        ret,img = video.read()
        frame = cv2.flip(img,1)
        frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)


        if results.multi_hand_landmarks:
            # passing the result of the frame to detect if there is multiple hands
            for hand_landmarks,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                hand = handedness.classification[0].label
                h,w,_ = frame.shape
                if hand == 'Left':
                        if finger_open(hand_landmarks):
                            cen_x,cen_y = get_center(hand_landmarks,w,h)
                            cv2.circle(frame, (cen_x, cen_y), 10, (0, 0, 255), 2)
                            
                            if prev_x is not None:
                                if (cen_x - prev_x) > movement_threshold and (time.time() - last_time > cooldown):
                                    print("DELETE WORD TRIGGERED!")
                                    last_time = time.time()

                            prev_x = cen_x
                if hand == 'Right':
                        if finger_open(hand_landmarks):
                            cen_x,cen_y = get_center(hand_landmarks,w,h)
                            cv2.circle(frame, (cen_x, cen_y), 10, (0, 0, 255), 2)
                            
                            if prev_x is not None:
                                if (prev_x - cen_x) > movement_threshold and (time.time() - last_time > cooldown):
                                    print("DELETE UNDO TRIGGERED!")
                                    last_time = time.time()

        cv2.imshow('MediaPipe Hands',frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
cv2.destroyAllWindows()