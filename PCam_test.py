
import numpy as np 
import pandas as pd
from datetime import datetime

from keras.models import Sequential
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense,Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy
from keras import regularizers, optimizers
from keras.optimizers import Adam

from keras.applications import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.nasnet import NASNetMobile


from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau

import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (20,10)

def append_ext(fn):
    return fn+".tif"


import os
print(os.listdir("input"))
import matplotlib


# import the necessary packages
import keras
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
import random
import shutil



from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import save_img
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
from PIL import Image
from keras.optimizers import Adagrad

traindf=pd.read_csv("input/train_labels.csv",dtype=str)
train_size = 180000
traindf = traindf.sort_values(by=['label','id'])
traindf = traindf.iloc[:int(train_size/2)].append(traindf.iloc[-int(train_size/2):])
traindf["id"]=traindf["id"].apply(append_ext)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

dataset = traindf
dataset_copy = dataset.copy()
train_set = dataset_copy.sample(frac=0.99, random_state=0)
test_set = dataset_copy.drop(train_set.index)
train_set['label'].value_counts()

test_set['label'].value_counts()

test_generator=test_datagen.flow_from_dataframe(
                                                dataframe=test_set,
                                                directory="input/test/",
                                                x_col="id",
                                                y_col="label",
                                                batch_size=128,
                                                seed=42,
                                                shuffle=False,
                                                class_mode="binary",
                                                target_size=(96, 96)
)

def auc(y_true, y_pred):
    """ROC AUC metric evaluator"""
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
dependencies = {
    'auc': auc
}
model = load_model('DenseNet-final.h5',custom_objects=dependencies)

predPyCam = model.predict_generator(test_generator,verbose=2)

predictions = predPyCam.copy()

predictions = np.rint(predictions)

print(classification_report(test_generator.classes, predictions,
			target_names=test_generator.class_indices.keys()))

cm = confusion_matrix(test_generator.classes, predictions)

sensitivity0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
sensitivity1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])
sensitivity = (sensitivity1+sensitivity0)/2
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total

f1 = 2*((sensitivity0*sensitivity1)/(sensitivity0+sensitivity1))
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
multiclass = cm

class_names = ['benign', 'metastatic']

fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_names)

plt.savefig('InceptionV3.png')

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_generator.classes, predictions)