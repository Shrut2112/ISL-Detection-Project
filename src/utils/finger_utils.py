import cv2
import mediapipe as mp
import time
import math

def finger_open(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    open = 0
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open += 1
    return open

def get_center(hand_landmarks, w, h):
    return int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)

def distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
