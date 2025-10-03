from flask import Flask, render_template, request, redirect, url_for, session
import os
import cv2
import numpy as np
from tensorflow import keras

# Flask app
app = Flask(__name__)
app.secret_key = 'supersecret'  # Required for session

# Load the trained model
model = keras.models.load_model("brain_tumor_model.keras", compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Label map
label_map = {
    0: 'pituitary',
    1: 'glioma',
    2: 'meningioma',
    3: 'notumor'
}

# Home page ('/login')
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['user'] = request.form['username']
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

# Index page
@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Read and preprocess
            image = cv2.imread(filepath)
            image_resized = cv2.resize(image, (128, 128))
            image_scaled = image_resized / 255.0
            image_input = np.reshape(image_scaled, (1, 128, 128, 3))

            # Predict
            prediction = model.predict(image_input)
            class_index = np.argmax(prediction)
            label = label_map[class_index]

            return render_template('predict.html', label=label, filename=file.filename)

    return render_template('predict.html', label=None)

# Serve uploaded files
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)





# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
