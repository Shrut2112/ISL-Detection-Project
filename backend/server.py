from flask import Flask, request,jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
from speech_to_text import GestureViewer
from Processing import HandGestureDetector
import speech_recognition as sr
import io
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

detector = HandGestureDetector()
gv = GestureViewer(socketio)

@app.route('/frame', methods=['POST'])
def process_frame():

    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1]) #gives binary byte
    np_img = np.frombuffer(img_data, np.uint8) #converts in np array
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)#decode array in cv2 image
    
    # Now you can run OpenCV/Mediapipe here
    
    result = detector.process_frame(frame)
    
    return jsonify(result)

r = sr.Recognizer()
@socketio.on("audio_utterence")
def speech_to_gesture(data):
    try:
        audio_bytes = bytes(data["buffer"])
        
        # Wrap in BytesIO
        webm_io = io.BytesIO(audio_bytes)
        
        # Convert from webm to wav using pydub
        audio_seg = AudioSegment.from_file(webm_io, format="webm")
        
        wav_io = io.BytesIO()
        audio_seg.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Now use SpeechRecognition
        with sr.AudioFile(wav_io) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        
        try:
            text = r.recognize_google(audio)
            print("You said:", text)
            gv.display_gesture(text)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            text = f"Speech service error: {e}"
        
        socketio.emit("recognized_text", {'text': text.lower()})
    
    except Exception as e:
        print("Error processing audio:", e)
        socketio.emit("recognized_text", {"text": ""})
            
@app.route('/speech-to-gesture',methods=['POST'])
def speech_to_gesture():
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
                        return {'recognized_text':text.lower()}
                    except:
                        print("Something Went Wrong")
                        break
                except sr.UnknownValueError:
                    print("Sorry, I could not understand.")
                    continue
                except sr.RequestError as e:
                    print("Speech service error:", e)
                    continue
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
