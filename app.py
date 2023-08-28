# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
from flask import Flask, request
import cv2
import os
import numpy as np
import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"


# prediction route
@app.route('/predict', methods=['POST'])
def predict():
    now = datetime.datetime.now()
    imagefile = request.files['imagefile']
    image_path = './Images/' +  now + imagefile.filename
    imagefile.save(image_path)
    print(f"""

    IMAGE PATH: {image_path}

    
    """)
    img = cv2.imread(image_path)
    resize = tf.image.resize(img, (256, 256))
    high_stroke = load_model(os.path.join('models', 'high_stroke.h5'))
    Park_scabies = load_model(os.path.join('models', 'Park_ScabiesModel.h5'))

    result = high_stroke.predict(np.expand_dims(resize/255, 0))
    result_1 = Park_scabies.predict(np.expand_dims(resize/255, 0))

    result_2 = {
        'stroke_results': result,
        'park_results': result_1
    }

    return result_2


if __name__ == '__main__':
    app.run(port=3000, debug=True)
