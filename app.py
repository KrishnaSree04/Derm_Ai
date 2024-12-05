from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

app = Flask(__name__)

# Load your trained model
model = load_model('my_model.h5')

# Define the skin disease classes
SKIN_CLASSES = {
    0: 'Actinic Keratoses',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

def classify_severity(score):
    if score < 0.33:
        return "Normal"
    elif score < 0.66:
        return "Mild"
    else:
        return "Severe"

@app.route('/')
def index():
    return render_template('health.html')

    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle login logic here
        return redirect(url_for('health'))  # Redirect to a health page or wherever you need
    return render_template('login_form.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup logic here
        return redirect(url_for('login'))  # Redirect to login after signup
    return render_template('signUp_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Ensure the static directory exists
        if not os.path.exists('static'):
            os.makedirs('static')
        
        filepath = os.path.join('static', file.filename)
        file.save(filepath)
        
        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        
        prediction = model.predict(img)
        pred_class = np.argmax(prediction, axis=1)[0]
        severity_score = prediction[0][pred_class]
        severity = classify_severity(severity_score)
        
        return render_template('health.html', prediction=SKIN_CLASSES[pred_class], severity=severity, image_file=file.filename)

if __name__ == '__main__':
    app.run(debug=True)






