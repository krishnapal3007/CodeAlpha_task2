import librosa
import numpy as np
import tensorflow as tf


MODEL_PATH = "emotion_model.keras"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_TIME_STEPS = 174


# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model(MODEL_PATH)

emotion_labels = np.load(
    "emotion_labels.npy",
    allow_pickle=True
)


# ==============================
# FEATURE EXTRACTION
# ==============================

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

    # Padding
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


# ==============================
# PREDICTION
# ==============================

file_path = input(
    "\nEnter path of WAV audio file: "
).strip().strip('"')


features = extract_features(
    file_path
)


prediction = model.predict(
    features,
    verbose=0
)


predicted_index = np.argmax(
    prediction
)

predicted_emotion = emotion_labels[
    predicted_index
]

confidence = (
    prediction[
        0,
        predicted_index
    ] * 100
)


# ==============================
# RESULT
# ==============================

print("\n===================================")
print("SPEECH EMOTION RESULT")
print("===================================")

print(
    f"Emotion    : {predicted_emotion}"
)

print(
    f"Confidence : {confidence:.2f}%"
)

print("\nAll Emotion Probabilities:")

for i, emotion in enumerate(
    emotion_labels
):

    print(
        f"{emotion:10s}: "
        f"{prediction[0][i] * 100:.2f}%"
    )

print("===================================")