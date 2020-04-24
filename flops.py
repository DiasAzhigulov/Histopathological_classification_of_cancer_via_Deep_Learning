# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:05:23 2020

@author: Qasymjomart
"""


import math

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.layers import Dense
from keras.callbacks import CSVLogger
from imutils import paths
from kerassurgeon.identify import get_apoz
from kerassurgeon import Surgeon
from keras import optimizers
from keras import models
from keras import layers
import keras.backend as K
import pandas as pd
import numpy as np
import tensorflow as tf
import os,csv



os.chdir('C:/Users/WW/Desktop/Cancer/ResNet101')

model = load_model('TMAD/ResNet101_pruning_25percent_TRAINED.h5')
#model = load_model('3_img_Best_model#100_ResNet101.h5')
model.summary()

def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()


    with graph.as_default():
        with session.as_default():

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops
        
print(get_flops(model))