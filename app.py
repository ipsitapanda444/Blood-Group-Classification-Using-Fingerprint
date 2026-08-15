from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model (replace 'model.h5' with your actual model file)
model = tf.keras.models.load_model('model.h5')

# Define blood group labels (modify based on your trained model)
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

def process_image(image_path):
    """ Preprocess the image for prediction """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img
from flask import Flask, render_template

app = Flask(__name__, template_folder='6TH SEM PROJECT')

@app.route('/')
def home():
    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'fingerprint' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['fingerprint']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Process the image and make a prediction
    img = process_image(file_path)
    prediction = model.predict(img)
    predicted_label = blood_groups[np.argmax(prediction)]
    
    return jsonify({'predicted_blood_group': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
