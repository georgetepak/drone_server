import os
import tempfile
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Параметры аудио должны совпадать с теми, на которых обучалась модель
SR = 16000
N_MFCC = 20
NUM_FRAMES = 49
MAX_DURATION = 1.0  # 1 секунда

MODEL_PATH = "drone_classifier.joblib"

app = Flask(__name__)
CORS(app)  # разрешаем запросы с другого домена (фронтенд на One)

# загружаем модель при старте
model = joblib.load(MODEL_PATH)


def extract_mfcc_from_audio(y, sr):
    """Превращает сигнал y в вектор признаков (1, 980)."""
    # Приводим к 1 секунде
    target_len = int(SR * MAX_DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # MFCC (20, T)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)

    # Приводим T к NUM_FRAMES
    current_frames = mfcc.shape[1]
    if current_frames < NUM_FRAMES:
        pad_width = NUM_FRAMES - current_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :NUM_FRAMES]

    mfcc_matrix = mfcc.T  # (49, 20)
    return mfcc_matrix.flatten().reshape(1, -1)  # (1, 980)


@app.route("/predict", methods=["POST"])
def predict():
    # Ожидаем файл в form-data с именем "audio"
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Временный файл (может быть .webm, .wav — librosa сам разберётся)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Загружаем через librosa
        y, sr = librosa.load(tmp_path, sr=SR, mono=True)
    except Exception as e:
        os.remove(tmp_path)
        return jsonify({"error": f"Cannot read audio: {e}"}), 500

    os.remove(tmp_path)

    X = extract_mfcc_from_audio(y, sr)
    proba = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])

    label = "drone" if pred == 1 else "nodrone"
    prob_drone = float(proba[1])

    return jsonify({
        "label": label,
        "prob_drone": prob_drone
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Drone detector API"})


if __name__ == "__main__":
    # Railway задаёт PORT как переменную окружения
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
