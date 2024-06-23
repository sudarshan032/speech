import os
import importlib.util
from flask import Flask, render_template, request, jsonify
import threading
import numpy as np
import sounddevice as sd
import wavio

# Importing modules from files
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_modules_from_folder(folder_path):
    imported_functions = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.py'):
            module_name = file_name[:-3]
            module_path = os.path.join(folder_path, file_name)
            module = import_module_from_file(module_name, module_path)
            for name in dir(module):
                if callable(getattr(module, name)):
                    imported_functions[name] = getattr(module, name)
    return imported_functions

folder_path = 'modules'
imported_functions = load_modules_from_folder(folder_path)

# Flask app setup
imgFolder = os.path.join('static', 'assets')
app = Flask(__name__)

FILE_TYPES = set(["wav", "WAV"])
recording = False
record_thread = None
record_data = []
output_file = "output.wav"

# Audio recording functions
def record_audio(filename, fs):
    global recording, record_data
    recording = True
    record_data = []
    try:
        print("Recording...")
        with sd.InputStream(samplerate=fs, channels=2, dtype='int16') as stream:
            while recording:
                data, _ = stream.read(1024)
                record_data.append(data)
    except Exception as e:
        print(f"Error recording audio: {e}")

    if record_data:
        audio_data = np.concatenate(record_data, axis=0)
        wavio.write(filename, audio_data, fs, sampwidth=2)
        print(f"Audio saved as {filename}")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/team")
def team():
    return render_template('team.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/vld", methods=["GET","POST"])
def vld():
    if request.method == "POST":
        if "file" not in request.files:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('vld.html', img_gif=buf)
        file = request.files["file"]
        if file.filename == "":
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('vld.html', img_gif=buf)
        if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
            print("Current Working Directory:", os.getcwd())
            buf, buf1 = imported_functions['Voice_Liveness_Detection'](file)
            return render_template('vld.html', img_gif=buf, emotion_name=buf1)
        else:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('vld.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('vld.html', img_gif=buf)

@app.route("/add", methods=["GET","POST"])
def add():
    if request.method == "POST":
        if "file" not in request.files:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('add.html', img_gif=buf)
        file = request.files["file"]
        if file.filename == "":
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('add.html', img_gif=buf)
        if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
            print("Current Working Directory:", os.getcwd())
            buf, buf1 = imported_functions['Audio_Deepfake_Detection'](file)
            return render_template('add.html', img_gif=buf, emotion_name=buf1)
        else:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('add.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('add.html', img_gif=buf)

@app.route("/dys", methods=["GET","POST"])
def dys():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "" and '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
                print("Current Working Directory:", os.getcwd())
                buf = imported_functions['dysarthric_asr'](file)
                return render_template('dys.html', img_gif=buf)
        elif "record" in request.form and "file_path" in request.form:
            print("Using recorded audio for classification.")
            file_path = request.form["file_path"]
            buf = imported_functions['dysarthric_asr'](file_path)
            return render_template('dys.html', img_gif=buf)
        else:
            buf = os.path.join(imgFolder, 'blank.png')
            return render_template('dys.html', img_gif=buf)
    buf = os.path.join(imgFolder, 'blank.png')
    return render_template('dys.html', img_gif=buf)

@app.route("/emotions", methods=["GET","POST"])
def emotions():
    if request.method == "POST":
        if "file" not in request.files:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('emotions.html', img_gif=buf)
        file = request.files["file"]
        if file.filename == "":
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('emotions.html', img_gif=buf)
        if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
            print("Current Working Directory:", os.getcwd())
            buf, buf1 = imported_functions['speech_emotion_recogition'](file)
            return render_template('emotions.html', img_gif=buf, emotion_name=buf1)
        else:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('emotions.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('emotions.html', img_gif=buf)

@app.route("/infant", methods=["GET","POST"])
def infant():
    if request.method == "POST":
        if "file" not in request.files:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('infant.html', img_gif=buf)
        file = request.files["file"]
        if file.filename == "":
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('infant.html', img_gif=buf)
        if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
            print("Current Working Directory:", os.getcwd())
            buf, buf1 = imported_functions['INFANT_CRY_CLASSIFICATION'](file)
            return render_template('infant.html', img_gif=buf, emotion_name=buf1)
        else:
            buf = os.path.join(imgFolder, 'blank.png')
            buf1 = os.path.join(imgFolder, 'blank.png')
            return render_template('infant.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('infant.html', img_gif=buf)

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/elements")
def elements():
    return render_template('elements.html')

@app.route("/services")
def services():
    return render_template('services.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, record_thread
    if not recording:
        recording = True
        record_thread = threading.Thread(target=record_audio, args=(output_file, 16000))
        record_thread.start()
        return jsonify({"status": "recording"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    if recording:
        recording = False
        record_thread.join()
        return jsonify({"status": "finished"})
    return jsonify({"status": "not recording"})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
