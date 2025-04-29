from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'secret-key'  # Needed for flash messages

# Upload config
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Canny + Threshold function
def apply_canny(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == 'Otsu Threshold':
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'Adaptive Threshold':
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, blockSize=11, C=2)
    else:
        thresh = blur  # fallback

    edges = cv2.Canny(thresh, 50, 150)
    return thresh, edges

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or not request.files['file'].filename:
        flash('No file selected.')
        return redirect(url_for('index'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        method = request.form.get('method', 'Adaptive Threshold')
        thresh, edges = apply_canny(image, method)

        # Save intermediate and result images
        thresh_filename = 'thresh_' + filename
        edges_filename = 'edges_' + filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], thresh_filename), thresh)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], edges_filename), edges)

        return render_template('result.html',
                               filename=filename,
                               thresh_filename=thresh_filename,
                               result_filename=edges_filename,
                               method=method)
    else:
        flash('Invalid file type. Please upload a valid image.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
