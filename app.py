import os
import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODEL_PATH  = os.path.join(WEIGHTS_DIR, "Meso4_DF_full.keras")
MODEL_URL   = (
    "https://github.com/AikagraGupta/DetectIT/"
    "releases/download/v1.0.0/Meso4_DF_full.keras"
)

if not os.path.exists(MODEL_PATH):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"⏬ Downloading model to {MODEL_PATH} …")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded.")

model = load_model(MODEL_PATH, compile=False)
print("Loaded model from", MODEL_PATH)

def preprocess_image(file_stream):
    buf = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256,256)).astype("float32") / 255.0
    return np.expand_dims(img, 0)

def sample_video_frames(file_stream, samples=30):
    tmp_path = os.path.join(BASE_DIR, "temp_video.mkv")
    with open(tmp_path, "wb") as f:
        f.write(file_stream.read())
    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // samples)
    frames, idx, taken = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % interval == 0 and taken < samples:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (256,256)).astype("float32") / 255.0
            frames.append(np.expand_dims(resized, 0))
            taken += 1
        idx += 1
    cap.release()
    os.remove(tmp_path)
    return frames

@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify(error="no file"), 400
    img_arr = preprocess_image(request.files["file"])
    score = float(model.predict(img_arr, verbose=0)[0][0])
    return jsonify(label="real" if score>=0.5 else "deepfake", confidence=score)

@app.route("/predict-video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify(error="no file"), 400
    frames = sample_video_frames(request.files["file"], samples=30)
    scores = [float(model.predict(f, verbose=0)[0][0]) for f in frames]
    avg = float(np.mean(scores)) if scores else 0.0
    return jsonify(label="real" if avg>=0.5 else "deepfake", confidence=avg)

port = int(os.environ.get("PORT", 5000))
print(f"Starting server on port {port}")
app.run(host="0.0.0.0", port=port)
