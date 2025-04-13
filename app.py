from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from imageai.Prediction import ImagePrediction
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize ImageAI prediction
prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
prediction.loadModel()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform image prediction
        predictions, probabilities = prediction.predictImage(filepath, result_count=5)
        
        result = []
        for pred, prob in zip(predictions, probabilities):
            result.append(f"{pred} ({prob:.2f}%)")
        
        return render_template('index.html', 
                             uploaded_image=filename,
                             predictions=result)
    
    return 'Invalid file type', 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=False, host='0.0.0.0', port=5000)
