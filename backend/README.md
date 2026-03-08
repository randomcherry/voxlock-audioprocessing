# VoxLock - Real-time Voice Privacy & Processing Prototype

**Technovation Girls 2026 Project**

## Problem
Voice data is often recorded without consent or processed insecurely. VoxLock aims to provide real-time, on-device audio enhancement and transcription while preserving privacy.

## Solution
- Real-time microphone input using PyAudio
- Voice Activity Detection (Silero VAD)
- Noise reduction (Facebook Denoiser)
- Audio sharpening (Butterworth high-pass filter)
- Speech-to-Text (Google via SpeechRecognition)
- Frontend chat UI with Socket.IO for live transcription display

## Tech Stack
- Backend: Python, Flask, Flask-SocketIO
- Audio/ML: PyTorch (Silero VAD + Denoiser), librosa, numpy, SpeechRecognition, PyAudio
- Frontend: HTML + JavaScript + Socket.IO client

## How to Run
1. Clone repo
2. cd backend
3. Install dependencies:
   ```bash
   pip install flask flask-cors flask-socketio torch torchvision torchaudio pyaudio SpeechRecognition