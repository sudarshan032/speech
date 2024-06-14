import sounddevice as sd
import wavio

def record_audio(filename, duration, fs):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait() 
    print("Recording finished.")
    wavio.write(filename, audio_data, fs, sampwidth=2)
    print(f"Audio saved as {filename}")

filename = "output.wav"
duration = 10  
fs = 16000 

record_audio(filename, duration, fs)
