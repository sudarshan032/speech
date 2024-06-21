from flask import Flask, render_template, request , redirect
from main import *

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
    print("hello")
    if request.method == "POST":
            todo = request.form.get("todo")
            print(todo)
            if "file" not in request.files:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('vld.html', img_gif=buf)
            file = request.files["file"]
            if file.filename == "":
                buf = os.path.join(imgFolder, 'blank.png')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('vld.html', img_gif=buf)
            if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
                print("Current Working Directory:", os.getcwd())
                buf, buf1 = fun4(file)
                return render_template('vld.html', img_gif=buf, emotion_name=buf1)
            else:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('vld.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('vld.html', img_gif=buf)

@app.route("/add", methods=["GET","POST"])
def add():
    print("hello")
    if request.method == "POST":
            todo = request.form.get("todo")
            print(todo)
            if "file" not in request.files:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('add.html', img_gif=buf)
            file = request.files["file"]
            if file.filename == "":
                buf = os.path.join(imgFolder, 'blank.png')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('add.html', img_gif=buf)
            if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
                print("Current Working Directory:", os.getcwd())
                buf, buf1 = fun5(file)
                return render_template('add.html', img_gif=buf, emotion_name=buf1)
            else:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('add.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('add.html', img_gif=buf)

# @app.route("/dys")
@app.route("/dys", methods=["GET","POST"])
def dys():
    print("hello")
    if request.method == "POST":
            # todo = request.form.get("todo")
            # print(todo)
            if "file" not in request.files:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('dys.html', img_gif=buf)
            file = request.files["file"]
            if file.filename == "":
                buf = os.path.join(imgFolder, 'blank.png')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('dys.html', img_gif=buf)
            if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
                print("Current Working Directory:", os.getcwd())
                buf = fun3(file)
                return render_template('dys.html', img_gif=buf)
            else:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('dys.html', img_gif=buf)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('dys.html', img_gif=buf)
    # return render_template('dys.html')


@app.route("/emotions", methods=["GET","POST"])
def emotions():
    print("hello")
    if request.method == "POST":
            todo = request.form.get("todo")
            print(todo)
            if "file" not in request.files:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('emotions.html', img_gif=buf)
            file = request.files["file"]
            if file.filename == "":
                buf = os.path.join(imgFolder, 'blank.png')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('emotions.html', img_gif=buf)
            if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
                print("Current Working Directory:", os.getcwd())
                buf, buf1 = fun1(file)
                return render_template('emotions.html', img_gif=buf, emotion_name=buf1)
            else:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('emotions.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('emotions.html', img_gif=buf)

@app.route("/infant", methods=["GET","POST"])
def infant():
    print("hello")
    if request.method == "POST":
            todo = request.form.get("todo")
            print(todo)
            if "file" not in request.files:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('infant.html', img_gif=buf)
            file = request.files["file"]
            if file.filename == "":
                buf = os.path.join(imgFolder, 'blank.png')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('infant.html', img_gif=buf)
            if '.' in file.filename and (file.filename).rsplit('.', 1)[1] in FILE_TYPES:
                print("Current Working Directory:", os.getcwd())
                buf, buf1 = fun2(file)
                return render_template('infant.html', img_gif=buf, emotion_name=buf1)
            else:
                buf = os.path.join(imgFolder, 'R.gif')
                buf1 = os.path.join(imgFolder, 'blank.png')
                return render_template('infant.html', img_gif=buf, emotion_name=buf1)
    buf = os.path.join(imgFolder, 'blank.png')
    buf1 = os.path.join(imgFolder, 'blank.png')
    return render_template('infant.html', img_gif=buf)
    # return render_template('infant.html', img_gif=buf, emotion_name=buf1)


@app.route("/contact")
def contact():

    return render_template('contact.html')


@app.route("/elements")
def elements():

    return render_template('elements.html')


@app.route("/services")
def services():

    return render_template('services.html')


app.run(debug=True, threaded=True)
