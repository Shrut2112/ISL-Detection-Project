from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
from src.utils.speech_to_text import GestureViewer
from src.utils.Processing import HandGestureDetector
import speech_recognition as sr
import io
from pydub import AudioSegment
from pydub.utils import which
from tensorflow.keras.models import load_model #type: ignore
import logging

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- FFMPEG ------------------
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# ----------------- Flask -------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ----------------- Static Gesture -----------
detector = HandGestureDetector()
gv = GestureViewer(socketio)

# ----------------- Dynamic Model Setup ------
MODEL_PATH = '../models/Dynamic.h5'

# Hardcoded class labels (must match training order)
CLASSES = ["hello", "thanks", "bye", "good", "congrats"]

try:
    model = load_model(MODEL_PATH)
    logger.info(f"Dynamic model loaded successfully with classes: {CLASSES}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

SEQUENCE_LENGTH = 20   # frames per sequence
FEATURES = 1662       # features per frame
CONFIDENCE_THRESHOLD = 0.7

# ----------- STATIC GESTURE ROUTE -------------
@app.route('/frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = detector.process_frame(frame)
    return jsonify(result)

# ----------- DYNAMIC GESTURE ROUTE -------------
@app.route('/predict_dynamic', methods=['POST'])
def predict_dynamic():
    try:
        data = request.json.get('sequence')
        if not data:
            return jsonify({'error': 'No sequence data provided'}), 400

        if len(data) != SEQUENCE_LENGTH:
            return jsonify({'error': f'Sequence must be {SEQUENCE_LENGTH} frames, got {len(data)}'}), 400

        seq = np.array(data, dtype=np.float32).reshape(1, SEQUENCE_LENGTH, FEATURES)
        if np.any(np.isnan(seq)) or np.any(np.isinf(seq)):
            return jsonify({'error': 'Sequence contains NaN or infinite values'}), 400

        preds = model.predict(seq, verbose=0)[0]
        predicted_idx = int(np.argmax(preds))
        confidence = float(preds[predicted_idx])

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({'gesture': 'Unknown', 'confidence': confidence})

        predicted_class = CLASSES[predicted_idx]
        logger.info(f"Predicted {predicted_class} with confidence {confidence:.2f}")

        return jsonify({'gesture': predicted_class, 'confidence': confidence})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# ----------- SPEECH TO GESTURE ROUTE -------------
r = sr.Recognizer()

@socketio.on("audio_utterence")
def speech_to_gesture(data):
    try:
        audio_bytes = bytes(data["buffer"])
        webm_io = io.BytesIO(audio_bytes)

        audio_seg = AudioSegment.from_file(webm_io, format="webm")
        wav_io = io.BytesIO()
        audio_seg.export(wav_io, format="wav")
        wav_io.seek(0)

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

# ----------------- Main ---------------------
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
