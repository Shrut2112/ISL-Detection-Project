import os
import cv2
from PIL import Image, ImageSequence
from flask_socketio import SocketIO, emit
import string
import numpy as np
import speech_recognition as sr
import base64

class GestureViewer:
    def __init__(self,socketio):
        self.socketio = socketio
        self.gif_folder = "src/images/ISL_Gifs"
        self.letter_folder = "src/images/letters"
        self.isl_gif = ['all the best', 'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
                'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
                'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
                'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
                'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
                 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
                'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
                'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call an ambulance', 'please call me later',
                'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
                'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
                'what are you doing', 'what is the problem', 'what is todays date', 'what is your age', 'what is your father do', 'what is your job',
                'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
                'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
                'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
                'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
                'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
                'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
                'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
                'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',  
                'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy']  # faster lookup

    
    def prepare_frame(self,file_path,types,delay=500):
        with open(file_path,"rb") as f:
            file_data = base64.b64encode(f.read()).decode("utf-8")
        return {'type':types,'data':file_data,'delay':delay}

    def display_gesture(self, text):
        
        text = text.lower()
        for c in string.punctuation:
            text = text.replace(c, "")
        words = text.split()
        i = 0
        sequence = []
        while i < len(words):
            matched = False
            for length in range(min(5, len(words) - i), 0, -1):
                phrase = " ".join(words[i:i + length])
                if phrase in self.isl_gif:
                    gif_path = os.path.join(self.gif_folder, f"{phrase}.gif")
                    if os.path.exists(gif_path):
                        sequence.append(self.prepare_frame(gif_path,'gif'))
                    matched = True
                    
                    i += length
                    break

            if not matched:
                for letter in words[i]:
                    if letter != " ":
                        letter_path = os.path.join(self.letter_folder, f"{letter}.jpg")
                        if os.path.exists(letter_path):
                            sequence.append(self.prepare_frame(letter_path,'letter'))
                   
                i += 1
        self.socketio.emit("gesture_sequence", sequence)

    def speech_to_gesture(self):
        """Listen from microphone and display gesture in real time."""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Speak now...")
            r.adjust_for_ambient_noise(source)
            while True:
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)
                    print("You said:", text)
                    if text.lower() == 'goodbye':
                        break
                    self.display_gesture(text)
                except sr.UnknownValueError:
                    print("Sorry, I could not understand.")
                    continue
                except sr.RequestError as e:
                    print("Speech service error:", e)
                    continue
                except Exception as e:
                    print("Something went wrong:", e)
                    break



