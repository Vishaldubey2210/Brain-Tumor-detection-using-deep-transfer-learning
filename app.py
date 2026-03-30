import numpy as np
from PIL import Image
import cv2
import gradio as gr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Use SAME architecture as training
base_model = VGG19(weights=None, include_top=False, input_shape=(240,240,3))

x = base_model.output
x = Flatten()(x)
x = Dense(4608, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1152, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = Model(base_model.input, output)

# Load your trained weights
model.load_weights("model_weights/vgg19_model_03.h5")

def predict(img):
    img = cv2.resize(img, (240,240))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)
    return "No Brain Tumor" if np.argmax(result)==0 else "Yes Brain Tumor"

gr.Interface(fn=predict, inputs="image", outputs="text").launch()