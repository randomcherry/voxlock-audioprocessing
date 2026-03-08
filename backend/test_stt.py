import speech_recognition as sr

r = sr.Recognizer()

print("Say something... (speak for 5 seconds)")
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=1)  # ← helps with background noise
    audio = r.listen(source, timeout=6, phrase_time_limit=8)

try:
    print("Sending to Google...")
    text = r.recognize_google(audio)
    print("You said:", text)
except sr.UnknownValueError:
    print("Google could not understand audio")
except sr.RequestError as e:
    print(f"Google API request failed: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__} - {e}")