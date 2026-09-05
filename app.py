from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import tensorflow as tf


# ==========================================
# APP CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    )
)


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "emotion_model.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "emotion_labels.npy"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_TIME_STEPS = 174


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading emotion model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

emotion_labels = np.load(
    LABEL_PATH,
    allow_pickle=True
)

print("Model loaded successfully!")


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Delta
    delta = librosa.feature.delta(
        mfcc
    )

    # Delta-Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine
    features = np.concatenate(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=0
    )

    # Normalize
    mean = np.mean(
        features,
        axis=1,
        keepdims=True
    )

    std = np.std(
        features,
        axis=1,
        keepdims=True
    )

    features = (
        features - mean
    ) / (
        std + 1e-8
    )

    # Padding / Truncation
    if features.shape[1] < MAX_TIME_STEPS:

        pad_width = (
            MAX_TIME_STEPS
            - features.shape[1]
        )

        features = np.pad(
            features,
            (
                (0, 0),
                (0, pad_width)
            ),
            mode="constant"
        )

    else:

        features = features[
            :,
            :MAX_TIME_STEPS
        ]

    # Time × Features
    features = features.T

    # Batch dimension
    features = np.expand_dims(
        features,
        axis=0
    )

    return features


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "audio" not in request.files:

        return render_template(
            "index.html",
            error="Please upload an audio file."
        )

    file = request.files["audio"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an audio file."
        )

    # Save file
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(file_path)

    try:

        # Extract features
        features = extract_features(
            file_path
        )

        # Prediction
        prediction = model.predict(
            features,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction[0]
        )

        emotion = emotion_labels[
            predicted_index
        ]

        confidence = float(
            prediction[
                0,
                predicted_index
            ] * 100
        )

        # All probabilities
        probabilities = []

        for i, label in enumerate(
            emotion_labels
        ):

            probabilities.append({
                "emotion": label,
                "probability": round(
                    float(
                        prediction[0][i] * 100
                    ),
                    2
                )
            })

        # Sort highest first
        probabilities.sort(
            key=lambda x: x["probability"],
            reverse=True
        )

        return render_template(
            "index.html",
            emotion=emotion,
            confidence=round(
                confidence,
                2
            ),
            probabilities=probabilities
        )

    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        return render_template(
            "index.html",
            error=f"Prediction failed: {str(e)}"
        )

    finally:

        # Delete uploaded file
        if os.path.exists(file_path):

            try:
                os.remove(file_path)
            except:
                pass


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )