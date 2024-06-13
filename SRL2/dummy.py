import sounddevice as sd
import wavio

def record_audio(filename, duration, fs):
    """
    Records an audio file and saves it.

    Parameters:
    filename (str): The name of the file to save the recording.
    duration (int): The duration of the recording in seconds.
    fs (int): The sampling frequency.

    Returns:
    None
    """
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    # Save the recording as a WAV file
    wavio.write(filename, audio_data, fs, sampwidth=2)
    print(f"Audio saved as {filename}")

# Parameters
filename = "output.wav"
duration = 10  # Duration in seconds
fs = 44100  # Sampling frequency

# Record and save the audio
record_audio(filename, duration, fs)
