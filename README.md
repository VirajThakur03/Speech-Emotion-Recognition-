# Speech-Emotion-Recognition-
Speech Emotion Recognition Web App
A Flask-based web application for real-time and file-based speech emotion recognition using deep learning. This project classifies emotions from audio input, either through live microphone recording or uploaded audio files.

# Features-

🔒 Login Screen for authentication

🌐 Web-based UI with a dark blue & black modern theme

🎤 Real-Time Audio Recording with playback preview

📁 Upload Audio Files (.wav) for emotion analysis

🧠 Deep Learning Model Integration for emotion classificatio

💬 Text-based Emotion Feedback

📦 Flask backend for API handling and prediction

☁️ Server-hostable for deployment on platforms like Heroku, Render, etc.

# Project Structure-

speech_emotion_recognition/
│
├── static/                  # CSS, JS, and UI assets
├── templates/               # HTML templates
│   ├── index.html           # Login page
│   └── home.html            # Main dashboard page
│
├── model/                   # Pre-trained model and training code
│   └── emotion_model.h5     # Trained emotion recognition model
│
├── app.py                   # Main Flask application
├── utils.py                 # Audio processing and model functions
├── requirements.txt         # Python dependencies
└── README.md                # Project overview

# Installation-
git clone https://github.com/yourusername/speech_emotion_recognition.git
cd speech_emotion_recognition

#Create virtual environment

python -m venv venv

source venv/bin/activate   # On Windows: venv\Scripts\activate

#Install dependencies

pip install -r requirements.txt

# Running the app-
python app.py

Visit http://127.0.0.1:5000/ in your browser.
