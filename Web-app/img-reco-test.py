#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 18:17:24 2018

@author: vivekkalyanarangan
"""

# For Genrating test images
#from PIL import Image
#from keras.datasets import mnist
#import numpy as np
#
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#for i in np.random.randint(0, 10000+1, 10):
#    arr2im = Image.fromarray(X_train[i])
#    arr2im.save('{}.png'.format(i), "PNG")

from keras.models import load_model
from PIL import Image
import numpy as np
from flasgger import Swagger
import matplotlib.pyplot as plt

from flask import render_template
from flask import Flask, request, redirect
##################    New code   ########################
from tensorflow import keras
import tensorflow as tf
import numpy as np
import logging as log

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)
##########################################################
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

##########################################################

app = Flask(__name__)
swagger = Swagger(app)

#model = load_model('model-kuat.h5')
modelBM = load_model('DenseNet121.h5')
modelBM._make_predict_function()
modelM = load_model('Malignant-sub-type-DenseNet121.h5')
modelM._make_predict_function()



@app.route('/', methods=['GET', 'POST'])
def index():
    user = {'username': 'Miguel'}
    if request.method == "POST":

        if request.files:
            try:
                with session.as_default():
                    with session.graph.as_default():
                        im = Image.open(request.files['image'])
                        #############################################

                        #############################################
                        classType = request.form['classification-type']
                        im = im.resize((224,224))
                        img_tensor = image.img_to_array(im)
                        img_tensor = np.expand_dims(img_tensor, axis=0)
                        img_tensor /= 255.
                    
                        if classType == "B-M":
                            result = np.argmax(modelBM.predict(img_tensor))

                            if (result == 1):
                                result = "Malignant"
                            else:
                                result = "Benign"
                        
                            result2 = modelBM.predict_proba(img_tensor)
                            classTypeStr = "Benign/malignant breast cancer"
                            

                        if classType == "M-sub":
                            result = np.argmax(modelM.predict(img_tensor))

                            if (result == 0):
                                result = "Carcinoma (DC)"
                            elif (result == 1):
                                result = "Lobular carcinoma (LC)"
                            elif (result == 2):
                                result = "Mucinous carcinoma (MC)"
                            else:
                                result = "Papillary carcinoma (PC)"

                        
                            result2 = modelM.predict_proba(img_tensor)

                            classTypeStr = "Malignant breast cancer sub-type"

                        #############################################
                        return render_template('result.html',  result=result, result2=result2[0], classType=classTypeStr)
            except:
                print("ERRRORRRRR!!!!")

    if request.method == 'GET':
        return render_template('index.html', title='Home', user=user)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)