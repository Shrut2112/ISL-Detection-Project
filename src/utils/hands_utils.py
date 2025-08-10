import numpy as np
import copy
import itertools

def get_both_hands(results,frame):
    # empty list of 42 dim 21 for x and 21 for y
    empty_hand = [0.0]*63
    # indicator for presence of left and right hand
    left_hand_pres = 0
    right_hand_pres = 0
    # copying empty_hand for passing values not reference
    left_hand = empty_hand.copy()
    right_hand = empty_hand.copy()
    # each hand is get seprate hand_landmarks so used loop to get both hand if there
    # handedness tells if hand is left or right
    for hand_landmarks,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
        landmark_list = get_landmark_list(hand_landmarks,frame)
        preprocess_list = preprocess(landmark_list)

        hand_type = handedness.classification[0].label

        if hand_type == 'Left':
            left_hand_pres = 1
            left_hand = preprocess_list
        if hand_type == 'Right':
            right_hand_pres = 1
            right_hand = preprocess_list

    #created row structure to append in csv
    row = [left_hand_pres,*left_hand,right_hand_pres,*right_hand]

    return row
    

def get_landmark_list(landmarks,image):
    #shape of frame
    h,w = image.shape[0],image.shape[1]

    landmark_points = []
    for lm in landmarks.landmark:

        landmark_x = lm.x
        landmark_y = lm.y
        landmark_z = lm.z

        #appending in landmark_points create 2d list
        landmark_points.append([landmark_x,landmark_y,landmark_z])

    return landmark_points

def preprocess(landmarks):
    #deep_copy to prevent reference 
    temp_landmark = copy.deepcopy(landmarks)
    #base to create realtive positions
    base_x,base_y = 0,0
    for index,landmark_pt  in enumerate(temp_landmark):

        if index == 0:
            #taking base hand position
            base_x,base_y,base_z = landmark_pt[0],landmark_pt[1],landmark_pt[2]

        temp_landmark[index][0] = temp_landmark[index][0] - base_x
        temp_landmark[index][1] = temp_landmark[index][1] - base_y
        temp_landmark[index][2] = temp_landmark[index][2] - base_z

    # converting 2d list ot 1d using itertools
    list_landmark = list(itertools.chain.from_iterable(temp_landmark))
    
    #getting max_values for normalization
    max_value = max(list(map(abs,list_landmark)))

    def normalized(n):
        return n/max_value
    #normalizing the list
    temp_list = list(map(normalized,list_landmark))
    return temp_list