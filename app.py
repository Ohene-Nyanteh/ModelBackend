# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
from flask import Flask, request
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"


# prediction route
@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['']
    image_path = './Images/' + imagefile.filename
    imagefile.save(image_path)

    return ""


if __name__ == '__main__':
    app.run(port=3000, debug=True)
