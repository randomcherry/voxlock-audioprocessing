import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=48000, input=True, frames_per_buffer=1024)

print("Listening for 10 seconds... Speak now!")
start = time.time()
while time.time() - start < 10:
    data = stream.read(1024, exception_on_overflow=False)
    audio = np.frombuffer(data, dtype=np.float32)
    max_amp = np.max(np.abs(audio))
    print(f"Amplitude: {max_amp:.6f}")
    time.sleep(0.5)

stream.stop_stream()
stream.close()
p.terminate()
print("Done.")