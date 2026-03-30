import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2

# Load a lightweight pretrained architecture (no weights to avoid extra memory/download)
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(240,240,3))

# Build custom classifier on top
x = base_model.output
x = Flatten()(x)                  # convert feature maps → vector
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)               # reduce overfitting
output = Dense(2, activation='softmax')(x)  # 2 classes

# Final model
model_03 = Model(base_model.input, output)

# Load trained weights (must match this architecture)
model_03.load_weights("model_weights/vgg19_model_03.h5")

# Flask app init
app = Flask(__name__)

# Convert prediction index → label
def get_className(classNo):
    return "No Brain Tumor" if classNo == 0 else "Yes Brain Tumor"

# Preprocess image + run prediction
def getResult(img_path):
    img = cv2.imread(img_path)                      # read image
    img = Image.fromarray(img, 'RGB')               # convert format
    img = img.resize((240, 240))                    # resize to model input
    img = np.array(img) / 255.0                     # normalize
    img = np.expand_dims(img, axis=0)               # add batch dimension

    result = model_03.predict(img)                  # inference
    return np.argmax(result, axis=1)                # class index

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload + prediction
@app.route('/predict', methods=['POST'])
def upload():
    file = request.files['file']

    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", secure_filename(file.filename))
    file.save(file_path)

    # Predict
    prediction = getResult(file_path)
    result = get_className(prediction[0])

    return render_template('result.html', result=result)

# Run server (Render compatible)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)