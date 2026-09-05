# 🎙️ Speech Emotion Recognition using Deep Learning

A deep learning based Speech Emotion Recognition (SER) system that analyzes human speech audio and predicts the emotion expressed in the voice.

The project uses **MFCC, Delta and Delta-Delta audio features** along with a **CNN + Bidirectional LSTM** neural network to classify speech into 8 different emotions.

---

## 🚀 Features

- 🎤 Speech emotion classification from WAV audio
- 🧠 CNN + BiLSTM deep learning model
- 🎵 MFCC + Delta + Delta-Delta feature extraction
- 📊 Emotion probability visualization
- 🌐 Professional Flask web interface
- ⚡ Real-time prediction from uploaded audio
- 💾 Trained model saved in `.keras` format

---

## 😊 Supported Emotions

The model can recognize the following 8 emotions:

1. 😐 Neutral
2. 😌 Calm
3. 😊 Happy
4. 😢 Sad
5. 😠 Angry
6. 😨 Fearful
7. 🤢 Disgust
8. 😲 Surprised

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Scikit-learn
- Flask
- SoundFile
- HTML
- CSS
- JavaScript

---

## 📂 Project Structure

```text
Speech_Emotion_Recognition/
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── app.py
├── train.py
├── predict.py
│
├── emotion_model.keras
├── emotion_labels.npy
│
├── requirements.txt
├── README.md
└── .gitignore