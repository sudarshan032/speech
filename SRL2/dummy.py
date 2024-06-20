import sounddevice as sd
import wavio
filename = "output.wav"
duration = 10  
fs = 16000 
def record_audio(filename, duration, fs):
    try:
        print("Recording...")
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
        sd.wait()
        print("Recording finished.")
        wavio.write(filename, audio_data, fs, sampwidth=2)
        print(f"Audio saved as {filename}")
    except Exception as e:
        print(f"Error recording audio: {e}")