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


traindf=pd.read_csv("input/train_labels.csv",dtype=str)
train_size = 180000
traindf = traindf.sort_values(by=['label','id'])
traindf = traindf.iloc[:int(train_size/2)].append(traindf.iloc[-int(train_size/2):])
testdf=pd.read_csv("input/sample_submission.csv",dtype=str)
traindf["id"]=traindf["id"].apply(append_ext)
testdf["id"]=testdf["id"].apply(append_ext)
datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)


B_size = 128

train_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="input/train/",
                                            x_col="id",
                                            y_col="label",
                                            subset="training",
                                            batch_size=B_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(96, 96)
)

valid_generator=datagen.flow_from_dataframe(
                                            dataframe=traindf,
                                            directory="input/train/",
                                            x_col="id",
                                            y_col="label",
                                            subset="validation",
                                            batch_size=B_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(96, 96)
)

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
                                                dataframe=testdf,
                                                directory="input/test/",
                                                x_col="id",
                                                y_col=None,
                                                batch_size=B_size,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(96, 96)
)


train_generator.n//train_generator.batch_size

def auc(y_true, y_pred):
    """ROC AUC metric evaluator"""
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def make_model(model_choice, model_name, input_tensor):
    '''Function to create a model
    Input:
    - model_choice, for ex: VGG19(include_top=False, input_tensor=input_tensor)
    - model_name, (str), name that will be given to the model in tensorboard
    
    Output:
    - model made with keras.model.Model'''
    
    base_model = model_choice
    x = base_model(input_tensor)
    out = Flatten()(x)
    out = Dense(1, activation="sigmoid")(out)
    model = Model(input_tensor, out)
    
    #The only callback we will use is TensorBoard, we could use early stopping or modifying the learning rate
    #but we wanted to compare the models as they were, with the same parameters for each.
    tensorboard=TensorBoard(log_dir = './logs/{}'.format(model_name),
                            histogram_freq=0,
                            batch_size=B_size,
                            write_graph=True,
                            write_grads=True,
                            write_images=False)
    
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['accuracy', auc])
    model.summary()
    
    history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    callbacks=[tensorboard])
    

    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title(model_name +  ' Model AUC')
    plt.legend([model_name +  ' Training',model_name +  ' Validation'])
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    
    return model


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



input_shape = (96, 96, 3)

ResNet50_model=ResNet50(include_top=False, input_tensor=None, weights='imagenet', input_shape = input_shape)
Rx = ResNet50_model.output
Rx = Flatten()(Rx)
prediction = Dense(1, activation="sigmoid")(Rx)
Rmodel = Model(ResNet50_model.input, prediction)

ResNet50_tensorboard = TensorBoard(log_dir = './logs/{}'.format('ResNet50'),
                                            histogram_freq=0,
                                            batch_size=B_size,
                                            write_graph=True,
                                            write_grads=True,
                                            write_images=False)

Rmodel.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['accuracy', auc])

history = Rmodel.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=10,
                    callbacks=[ResNet50_tensorboard]
)
end = datetime.now()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ResNet50 model accuracy')
plt.legend(['ResNet50_training','ResNet50_validation'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('acc-val')

plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('ResNet50 model accuracy')
plt.legend(['ResNet50_training','ResNet50_validation'])
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.savefig('auc-val_auc')

Rmodel.save('kuat.h5')