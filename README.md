# Brain Tumor Detection using Deep Transfer Learning

## Overview

This project is a Flask-based web app that detects brain tumors from MRI images using a VGG19-based deep learning model.

## Features

* Upload MRI images via web interface
* Predicts: **Tumor / No Tumor**
* Uses transfer learning (VGG19)

## Tech Stack

* Python, Flask
* TensorFlow / Keras
* OpenCV, NumPy, PIL

## Project Structure

```
├── app.py
├── model_weights/
├── uploads/
├── templates/
│   ├── index.html
│   └── result.html
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Usage

* Open: http://127.0.0.1:5000/
* Upload an MRI image
* View prediction

## Notes

* Model weights not included (use external storage)
* Add your `.h5` files in `model_weights/`

## License

For educational use only
