import os
import importlib.util
from flask import Flask, render_template, request

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


imgFolder = os.path.join('static', 'assets')
app = Flask(__name__)

FILE_TYPES = set(["wav", "WAV"])
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
                buf, buf1 = buf, buf1 = imported_functions['Voice_Liveness_Detection'](file)
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
            if "file" not in request.files:
                buf = os.path.join(imgFolder, 'blank.png')
                return render_template('dys.html', img_gif=buf)
            file = request.files["file"]
            if file.filename == "":
                buf = os.path.join(imgFolder, 'blank.png')
                return render_template('dys.html', img_gif=buf)
            if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
                print("Current Working Directory:", os.getcwd())
                buf = imported_functions['dysarthric_asr'](file)
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

if __name__=="__main__":
    app.run(debug=True, threaded=True)
