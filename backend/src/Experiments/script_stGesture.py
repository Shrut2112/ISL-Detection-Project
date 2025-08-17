import os
import cv2
from PIL import Image, ImageTk, ImageSequence
import time
import string
import numpy as np
import speech_recognition as sr

GIF_FOLDER = "src/images/ISL_Gifs"
LETTER_FOLDER = "src/images/letters"
isl_gif=['all the best', 'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
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
                'voice', 'wednesday', 'weight','please wait for sometime','what is your mobile number','what are you doing','are you busy']

WINDOW_NAME = "Gesture Viewer"
DISPLAY_SIZE = (600, 400)
current_frame = None

def fade_transition(duration=400, steps=10):
    """Creates a fade-to-black effect."""
    global current_frame
    black_frame = np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), dtype=np.uint8)
    for alpha in np.linspace(0, 1, steps):
        blended = cv2.addWeighted(current_frame, 1 - alpha, black_frame, alpha, 0)
        cv2.imshow(WINDOW_NAME, blended)
        if cv2.waitKey(int(duration / steps)) & 0xFF == 27:
            return False
    return True

def play_gif(path):
    global current_frame
    gif = Image.open(path)

    for frame in ImageSequence.Iterator(gif):
        frame_rgb = frame.convert("RGB").resize(DISPLAY_SIZE)
        frame_bgr = cv2.cvtColor(np.array(frame_rgb),cv2.COLOR_RGB2BGR)
        current_frame = frame_bgr
        cv2.imshow(WINDOW_NAME,frame_bgr)
        if cv2.waitKey(80) & 0xFF == 27:  # ESC to exit
            return False
    return True

def play_letter(path):
    global current_frame
    img = cv2.imread(path)
    img_resized = cv2.resize(img,DISPLAY_SIZE)
    current_frame = img_resized
    cv2.imshow(WINDOW_NAME,img_resized)
    if cv2.waitKey(500) & 0xFF == 27:  # ESC to exit early
        return False
    return True
    
def display_gesture(text):
    text = text.lower()
    for c in string.punctuation:
        text = text.replace(c, "")

    words = text.split()
    i = 0

    while i < len(words):
        matched = False

        # Try longest phrase first (e.g., 3 words, then 2 words, then 1)
        for length in range(min(5, len(words) - i), 0, -1):  
            phrase = " ".join(words[i:i+length])

            if phrase in isl_gif:
                gif_path = os.path.join(GIF_FOLDER, f"{phrase}.gif")
                if os.path.exists(gif_path):
                    if not play_gif(gif_path):
                        return
                matched = True
                fade_transition()
                i += length
                break

        # If no GIF phrase match, show the word letter-by-letter
        if not matched:
            for letter in words[i]:
                if letter != " ":
                    letter_path = os.path.join(LETTER_FOLDER, f"{letter}.jpg")
                    if os.path.exists(letter_path):
                        if not play_letter(letter_path):
                            return
                fade_transition()
            i += 1

    cv2.destroyAllWindows()

def Speech_to_gesture():
        r = sr.Recognizer()

        with sr.Microphone() as source:
            print("🎤 Speak now...")
            r.adjust_for_ambient_noise(source)
           
            while True:
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio)
                    print("You said: ",text)
                    
                    if  text.lower() =='goodbye':
                        break
                    try:
                        display_gesture(text)
                    except:
                        print("Something Went Wrong")
                        break
                except sr.UnknownValueError:
                    print("Sorry, I could not understand.")
                    continue
                except sr.RequestError as e:
                    print("Speech service error:", e)
                    continue

Speech_to_gesture()
