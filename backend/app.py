from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from audio_processing import process_audio

app = Flask(__name__)

# Enable CORS (allow frontend to connect)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Home route - serves index.html from templates/
@app.route('/')
def home():
    return render_template('index.html')

# Socket.IO connection event
@socketio.on('connect')
def handle_connect():
    print('Client connected via Socket.IO')
    emit('message', {'data': 'Connected to VoxLock backend!'})

# Echo test message
@socketio.on('message')
def handle_message(data):
    print('Received message:', data)
    emit('message', {'data': f'Echo: {data}'}, broadcast=True)

# Function to send transcription (call this from audio_processing.py)
def broadcast_transcription(text):
    socketio.emit('transcription', {'text': text})
    print(f"Broadcast transcription: {text}")

# Run the app with Socket.IO
if __name__ == '__main__':
    print("Starting VoxLock backend with Socket.IO...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)