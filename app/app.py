from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

app = Flask(__test__)


model = keras.models.load_model('model/test_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file.stream)
        img = img.resize((150, 150))  
        img_array = np.array(img) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        return f'Predicted class: {predicted_class}'

if __test__ == '__main__':
    app.run(host='0.0.0.0', port=5000)