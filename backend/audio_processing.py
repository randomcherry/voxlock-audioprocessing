import librosa
import numpy as np
import torch
import pyaudio
import time
import speech_recognition as sr
import io
import wave
import threading
import queue
from scipy.signal import butter, lfilter

# ───────────────────────────────────────────────
# 1. LOAD MODELS (only once at startup)
# ───────────────────────────────────────────────
print("Loading VAD model...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
print("Loading Denoiser...")
denoiser = torch.hub.load(
    'facebookresearch/denoiser',
    'dns64',
    force_reload=False,
    trust_repo=True
)
denoiser.eval()

# Speech recognizer
r = sr.Recognizer()

# ───────────────────────────────────────────────
# 2. CONFIGURATION
# ───────────────────────────────────────────────
MIC_RATE = 48000          # Microphone sample rate
MODEL_RATE = 16000        # Rate expected by VAD & Denoiser
CHUNK_48K = 1536          # Chunk size at 48kHz (~30 ms)
SILENCE_LIMIT = 3         # How many silent chunks before processing sentence
VAD_THRESHOLD = 0.25      # Lowered from 0.4 to detect speech more easily

# Queues & buffers
processing_queue = queue.Queue()     # Sends full sentences to background worker
sentence_buffer = []                 # Collects audio chunks during speech
is_recording = False
silence_counter = 0

# ───────────────────────────────────────────────
# 3. HELPER: Sharpen audio (high-pass filter)
# ───────────────────────────────────────────────
def sharpen_audio(data, cutoff=300, fs=MODEL_RATE):
    nyq = 0.5 * fs # Nyquist frequency
    b, a = butter(5, cutoff / nyq, btype='high', analog=False) # apply Butterworth filter
    sharpened = lfilter(b, a, data).astype(np.float32) # apply IIR filter
    return sharpened

# ───────────────────────────────────────────────
# 4. BACKGROUND WORKER THREAD (denoise → sharpen → STT → broadcast)
# ───────────────────────────────────────────────
def ai_worker():
    while True:
        full_audio = processing_queue.get()
        if full_audio is None:
            break

        print("[Worker] Processing sentence of length:", len(full_audio))

        # A. Denoise
        input_t = torch.from_numpy(full_audio).unsqueeze(0) # denoiser expects 2D (1,N)
        with torch.inference_mode():
            denoised_t = denoiser(input_t)
        output_np = denoised_t.squeeze(0).numpy()

        # B. Sharpen
        sharpened = sharpen_audio(output_np)

        # C. Transcribe (Google STT)
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)          # Google STT demands 16-bit
            wav_file.setframerate(MODEL_RATE)
            int_data = (sharpened * 32767).clip(-32768, 32767).astype(np.int16) # clamp to int16 range
            wav_file.writeframes(int_data.tobytes())

        byte_io.seek(0)
        with sr.AudioFile(byte_io) as source:
            try:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                if text.strip():
                    print(f"[STT]: {text}")
                    # Send to frontend via Socket.IO
                    broadcast_transcription(text)
            except sr.UnknownValueError:
                print("[STT] Could not understand audio")
            except sr.RequestError as e:
                print(f"[STT] Google API error: {e}")
            except Exception as e:
                print(f"[STT] Unexpected error: {e}")

        peak = np.max(np.abs(output_np))
        if peak > 0.05:
            output_np = (output_np / (peak + 1e-7)) * 0.8
            stream_out.write(output_np.tobytes())

        processing_queue.task_done()

# Start worker thread
threading.Thread(target=ai_worker, daemon=True).start()

# ───────────────────────────────────────────────
# 5. MAIN LISTENER LOOP (mic → VAD → buffer → queue)
# ───────────────────────────────────────────────
print("Initializing PyAudio...")
p = pyaudio.PyAudio()

# Show all input devices clearly
print("\n=== AVAILABLE MICROPHONES ===")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"[{i}] {dev['name']} | rate: {int(dev['defaultSampleRate'])}Hz | channels: {dev['maxInputChannels']}")

print("=============================\n")

# IMPORTANT: Change this number to match YOUR microphone from the list above
DEVICE_INDEX = 0   # ← Change to 1, 2, etc. if your mic is not index 0

print(f"Using microphone: [{DEVICE_INDEX}] {p.get_device_info_by_index(DEVICE_INDEX)['name']}")

# Open stream
stream_in = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=MIC_RATE,
    input=True,
    frames_per_buffer=CHUNK_48K,
    input_device_index=DEVICE_INDEX
)

stream_out = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=MODEL_RATE,
    output=True
)

print("\n>>> LISTENING STARTED! Speak now. <<<\n")
print("Watch for 'max amp' jumping when you speak (should be >0.02)\n")

try:
    while True:
        data = stream_in.read(CHUNK_48K, exception_on_overflow=False)
        print(f"Chunk read - size: {len(data)} bytes")

        audio_16k = np.frombuffer(data, dtype=np.float32)[::3].copy()
        max_amp = np.max(np.abs(audio_16k))
        print(f"Audio stats - max amplitude: {max_amp:.6f}  (speak to see this increase!)")

        if max_amp < 0.005:
            print(" → Chunk is very quiet/silent")

        # VAD
        with torch.inference_mode():
            speech_prob = vad_model(torch.from_numpy(audio_16k), MODEL_RATE).item()
        print(f"Speech probability: {speech_prob:.4f}")

        if speech_prob > VAD_THRESHOLD:
            is_recording = True
            sentence_buffer.append(audio_16k)
            silence_counter = 0
            print(" → SPEECH DETECTED! Starting to record...")
        elif is_recording:
            sentence_buffer.append(audio_16k)
            silence_counter += 1
            print(f"Silence counter: {silence_counter}/{SILENCE_LIMIT}")

            if silence_counter > SILENCE_LIMIT:
                print(" → Sentence finished → processing...")
                if sentence_buffer:
                    full_sentence = np.concatenate(sentence_buffer)
                    processing_queue.put(full_sentence)
                    print(f"Sent {len(full_sentence)} samples to worker")
                sentence_buffer = []
                is_recording = False
                silence_counter = 0

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"Error in loop: {e}")
finally:
    print("Closing audio streams...")
    stream_in.stop_stream()
    stream_in.close()
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
    processing_queue.put(None)  # stop worker thread

try:
    while True:
        try:
            data = stream_in.read(CHUNK_48K, exception_on_overflow=False)
            print(f"Chunk read OK - size: {len(data)} bytes")

            if len(data) == 0:
                print("WARNING: Empty chunk read")
                continue

            audio_16k = np.frombuffer(data, dtype=np.float32)[::3].copy()
            max_amp = np.max(np.abs(audio_16k))
            print(f"Audio data - shape: {audio_16k.shape}, max amp: {max_amp:.6f}")

            if max_amp < 0.005:
                print(" → Very quiet or silent chunk")

            # VAD
            with torch.inference_mode():
                vad_input = torch.from_numpy(audio_16k)
                speech_prob = vad_model(vad_input, MODEL_RATE).item()

            print(f"Speech prob: {speech_prob:.4f}")

            if speech_prob > VAD_THRESHOLD:
                is_recording = True
                sentence_buffer.append(audio_16k)
                silence_counter = 0
                print(" → SPEECH DETECTED! Recording...")
            elif is_recording:
                sentence_buffer.append(audio_16k)
                silence_counter += 1
                print(f"Silence counter: {silence_counter}/{SILENCE_LIMIT}")

                if silence_counter > SILENCE_LIMIT:
                    print(" → Sentence complete → sending to worker")
                    if sentence_buffer:
                        full_sentence = np.concatenate(sentence_buffer)
                        print(f"Full sentence length: {len(full_sentence)} samples")
                        processing_queue.put(full_sentence)
                    sentence_buffer = []
                    is_recording = False
                    silence_counter = 0

        except IOError as e:
            print("IOError in read:", e)
            continue

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"Main loop error: {e}")
finally:
    print("Closing streams...")
    stream_in.stop_stream()
    stream_in.close()
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
    processing_queue.put(None)  # stop worker

print("\n>>> Listening! Press Ctrl+C to stop. <<<\n")
print("Speak clearly and pause 3–4 seconds after speaking.\n")

try:
    while True:
        data = stream_in.read(CHUNK_48K, exception_on_overflow=False)
        print(f"Chunk received, length: {len(data)} bytes")  # debug

        audio_16k = np.frombuffer(data, dtype=np.float32)[::3].copy()
        print(f"Downsampled: shape={audio_16k.shape}, max_amp={np.max(np.abs(audio_16k)):.4f}")

        # Voice Activity Detection
        with torch.inference_mode():
            vad_input = torch.from_numpy(audio_16k)
            speech_prob = vad_model(vad_input, MODEL_RATE).item()

        print(f"Speech probability: {speech_prob:.3f}")

        if speech_prob > VAD_THRESHOLD:
            is_recording = True
            sentence_buffer.append(audio_16k)
            silence_counter = 0
            print("→ Speech detected! Recording...")
        elif is_recording:
            sentence_buffer.append(audio_16k)
            silence_counter += 1
            print(f"Silence counter: {silence_counter}/{SILENCE_LIMIT}")

            if silence_counter > SILENCE_LIMIT:
                print("→ Sentence complete → sending to worker")
                if sentence_buffer:
                    full_sentence = np.concatenate(sentence_buffer)
                    processing_queue.put(full_sentence)
                sentence_buffer = []
                is_recording = False
                silence_counter = 0

except KeyboardInterrupt:
    print("\nStopping...")
except Exception as e:
    print(f"Unexpected error in main loop: {e}")
finally:
    print("Closing streams...")
    stream_in.stop_stream()
    stream_in.close()
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
    processing_queue.put(None)  # stop worker