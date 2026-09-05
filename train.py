import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_PATH = "dataset"

SAMPLE_RATE = 22050

N_MFCC = 40

# MFCC + Delta + Delta-Delta
N_FEATURES = N_MFCC * 3

MAX_TIME_STEPS = 174


# ==========================================
# 2. EMOTION MAPPING
# ==========================================

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE
        )

        # ----------------------------------
        # MFCC
        # ----------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # ----------------------------------
        # Delta
        # ----------------------------------

        delta = librosa.feature.delta(
            mfcc
        )

        # ----------------------------------
        # Delta-Delta
        # ----------------------------------

        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # ----------------------------------
        # Combine
        # ----------------------------------

        features = np.concatenate(
            [
                mfcc,
                delta,
                delta2
            ],
            axis=0
        )

        # ----------------------------------
        # Normalize
        # ----------------------------------

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

        # ----------------------------------
        # Padding / Truncation
        # ----------------------------------

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

        # ----------------------------------
        # Convert:
        # features × time
        #
        # to:
        # time × features
        # ----------------------------------

        features = features.T

        return features

    except Exception as e:

        print(
            "Error processing:",
            file_path
        )

        print(e)

        return None


# ==========================================
# 4. LOAD DATASET
# ==========================================

X = []
y = []

print("\n===================================")
print("Loading RAVDESS Dataset...")
print("===================================\n")


for actor in sorted(
    os.listdir(DATASET_PATH)
):

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.isdir(actor_path):
        continue

    if not actor.startswith("Actor_"):
        continue

    print(
        "Processing:",
        actor
    )

    for filename in os.listdir(
        actor_path
    ):

        if not filename.lower().endswith(
            ".wav"
        ):
            continue

        file_path = os.path.join(
            actor_path,
            filename
        )

        # RAVDESS format:
        #
        # 03-01-05-01-02-01-12.wav
        #
        # Third value = emotion

        parts = filename.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code not in emotion_map:
            continue

        emotion = emotion_map[
            emotion_code
        ]

        features = extract_features(
            file_path
        )

        if features is not None:

            X.append(features)

            y.append(emotion)


# ==========================================
# 5. NUMPY CONVERSION
# ==========================================

X = np.array(X)

y = np.array(y)


print("\n===================================")
print("Dataset Loaded Successfully!")
print("===================================")

print(
    "X shape:",
    X.shape
)

print(
    "Total samples:",
    len(y)
)


# ==========================================
# 6. ENCODE LABELS
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

y_categorical = to_categorical(
    y_encoded
)


print("\nEmotion Classes:")

for index, emotion in enumerate(
    label_encoder.classes_
):

    print(
        index,
        "=",
        emotion
    )


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_categorical,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==========================================
# 8. BUILD CNN + BiLSTM MODEL
# ==========================================

model = Sequential([

    Input(
        shape=(
            MAX_TIME_STEPS,
            N_FEATURES
        )
    ),

    # ----------------------------------
    # CNN
    # ----------------------------------

    Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.25),

    # ----------------------------------
    # Second CNN
    # ----------------------------------

    Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.25),

    # ----------------------------------
    # BiLSTM
    # ----------------------------------

    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        )
    ),

    Dropout(0.35),

    Bidirectional(
        LSTM(
            64
        )
    ),

    Dropout(0.35),

    # ----------------------------------
    # Dense Layers
    # ----------------------------------

    Dense(
        128,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )

])


# ==========================================
# 9. COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


print("\n===================================")
print("Improved Model Summary")
print("===================================\n")

model.summary()


# ==========================================
# 10. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=1e-6,

    verbose=1

)


# ==========================================
# 11. TRAIN
# ==========================================

print("\n===================================")
print("Starting Improved Training...")
print("===================================\n")


history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=40,

    batch_size=32,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1

)


# ==========================================
# 12. EVALUATE
# ==========================================

loss, accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0

)


print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 13. SAVE MODEL
# ==========================================

model.save(
    "emotion_model.keras"
)


np.save(
    "emotion_labels.npy",
    label_encoder.classes_
)


print("\n===================================")
print("Training Completed Successfully!")
print("===================================")

print(
    "\nModel saved:"
)

print(
    "emotion_model.keras"
)

print(
    "\nLabels saved:"
)

print(
    "emotion_labels.npy"
)