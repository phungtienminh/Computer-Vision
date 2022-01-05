import flask
from flask import Flask, render_template, request
import os
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from models import load_model

model = load_model()


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def perform_prediction():
    file = request.files['file']
    savepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(savepath)

    # Make prediction
    image = Image.open(savepath)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    standardized_tensor = transform(image)
    standardized_tensor = standardized_tensor.unsqueeze(0)

    with torch.no_grad():
        prob = model(standardized_tensor)
        maxprob = prob.max().item()
        pos = prob.argmax(dim = 1)
        if pos == 0:
            predicted_label = 'BrownSpot'
        elif pos == 1:
            predicted_label = 'Healthy'
        elif pos == 2:
            predicted_label = 'Hispa'
        elif pos == 3:
            predicted_label = 'LeafBlast'
        else:
            predicted_label = 'Unknown'

    return render_template('result.html', filename = file.filename, confidence = maxprob, label = predicted_label)


if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 5000, debug = True)
