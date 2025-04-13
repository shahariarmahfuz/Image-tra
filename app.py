from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained model
model = MobileNetV2(weights='imagenet')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = analyze_image(filepath)
            return render_template('index.html', result=result)
    return render_template('index.html')

def analyze_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    confidence = decoded[2] * 100
    return f"ছবিতে সম্ভবত: {label} (নিশ্চয়তা: {confidence:.2f}%)"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
