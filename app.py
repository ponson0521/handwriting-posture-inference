import os
import time
import cv2 as cv
from flask import Flask, request, jsonify, render_template, flash, url_for, redirect, send_from_directory, Response, session
from werkzeug.utils import secure_filename
from mediapipe_draw import image_draw, video_draw
from datetime import datetime

# flask setup
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
UPLOAD_FOLDER = './static/storage'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024     # maximum filesize 16MB
nowtime = F"{time.localtime(time.time())[0]}-{time.localtime(time.time())[1]}-{time.localtime(time.time())[2]}"
labels = ["拳握法", "旋前抓握", "靜態抓握", "動態抓握", "指側抓握"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['POST'])    # simple upload function for image
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'filename' not in request.files:
            flash('不支援檔案格式')
            return redirect(url_for('/upload'))

        file = request.files['filename']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('未選擇檔案')
            return redirect(url_for('/upload'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            extension = filename.split(".")[1]
            if extension == "mp4":
                video_draw(file, filename)
                flash('上傳成功')
                return render_template("uldet.html", filename=filename, extension=extension)
            else:
                image, prediction = image_draw(file)
                cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), image)
                flash('上傳成功')
                return render_template("uldet.html", filename=filename, extension=extension, prediction=labels[prediction])
    return render_template("upload.html")

@app.route("/medijs")    # main function
def camera():
    return render_template("medijs.html")

@app.route('/axios', methods=['GET', 'POST'])    # use axios to upload image with results when using main function
def axios():
    if request.method == 'POST':
        file = request.files['file']
        prediction = request.form.get("anwser")
        leftright = request.form.get("leftright")
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return "success"
    
@app.route("/display/<filename>")    # show the uploaded image for /upload_file
def display(filename):
    return redirect(url_for('static', filename='storage/'+filename))

@app.route("/")    
def homepage():
    return render_template("index.html")

@app.route("/upload")    # image upload web page 
def upload():
    return render_template("upload.html")

@app.route("/product")   
def product():
    return render_template("product.html")

@app.route('/show_add_user')    # a form to record user info
def show_add_user():
    return render_template("show_add_user.html")

@app.route("/do_add_user", methods=['POST'])    # post the form
def do_add_user():
    name = request.form.get("name")
    gender  = request.form.get("gender")
    handedness = request.form.get("handedness")
    email = request.form.get("email")
    Date_birth = request.form.get("Date_birth")
    message= request.form.get("message")
    phone = request.form.get("phone")
    age = ((datetime.strptime(nowtime, "%Y-%m-%d")-datetime.strptime(Date_birth, "%Y-%m-%d"))//365).days
    session['age'] = age
    return render_template("medijs.html", age=age)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5521)
