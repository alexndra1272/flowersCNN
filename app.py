from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image

import numpy as np

app = Flask(__name__)


model = load_model('flowers_classifier.keras')

def preprocess_image(image):
    image = image.resize((256, 256))  # Redimensionar al tamaño requerido por el modelo

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = np.array(image)
    image = image / 255.0  # Normalizar los valores de píxeles al rango [0, 1]
    image = np.expand_dims(image, axis=0)  # Agregar dimensión de lote
    return image

def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions, axis=1)[0]
    class_names = ["Margaritas", "Girasoles", "Tulipanes"]
    class_name = class_names[class_index]
    probability = predictions[0, class_index]  # Probabilidad para la clase predicha
    percent = int(round(probability, 2) * 100)
    
    if percent < 80:
        return "No coincide con ninguna de las clases", 0
    return class_name, percent


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar que se está enviando un archivo de imagen en la solicitud POST
    if 'image' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo de imagen'})

    file = request.files['image']

    # Verificar que el archivo tiene un nombre
    if file.filename == '':
        return jsonify({'error': 'El archivo de imagen no tiene nombre'})

    try:
        # Abrir la imagen usando PIL
        image = Image.open(file)
        # Clasificar la imagen usando la función classify_image()
        class_name, confidence = classify_image(image)
        # Devolver el resultado como JSON
        return jsonify({'class': class_name, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)})