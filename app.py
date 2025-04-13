import os
from flask import Flask, request, render_template
from imageai.Detection import ObjectDetection
from werkzeug.utils import secure_filename

app = Flask(__name__)

# আপলোড ফোল্ডারের পাথ
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ইমেজ ডিটেকশন মডেলের পাথ
MODEL_PATH = 'yolov3.pt'  # আপনার ডাউনলোড করা মডেল ফাইলের পাথ দিন
OUTPUT_PATH = 'static/output'

# যদি আপলোড এবং আউটপুট ফোল্ডার না থাকে তবে তৈরি করুন
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# অবজেক্ট ডিটেকশন মডেল লোড করা হচ্ছে
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(MODEL_PATH)
detector.loadModel()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='কোন ফাইল আপলোড করা হয়নি')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='কোন ফাইল আপলোড করা হয়নি')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        output_filename = f"detected_{filename}"
        output_filepath = os.path.join(OUTPUT_PATH, output_filename)

        detections = detector.detectObjectsFromImage(input_image=filepath, output_image_path=output_filepath)

        detected_objects = [detection["name"] for detection in detections]

        return render_template('index.html', filename=output_filename, objects=detected_objects)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
    
