# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:56:01 2020

@author: dias.azhigulov
"""
import numpy as np
import os
from imutils import paths
from keras.models import load_model
import matplotlib.pyplot as plt
from kerassurgeon import identify
from kerassurgeon.operations import delete_channels,delete_layer
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

name = 'DenseNet121'
#TMAD dataset
os.chdir('D:\Capstone_results\DenseNet121')
model = load_model("0_AllLayer_model#"+str(100)+'_'+ name + ".h5")


#%%
#Conv2D layers
w1 = model.layers[0].layers[2].get_weights()[0]
w2 = model.layers[0].layers[9].get_weights()[0]
w3 = model.layers[0].layers[12].get_weights()[0]


conv_layer_weights = w1

#for i in range(len(conv_layer_weights)):
    weight = conv_layer_weights[i]
    weights_dict = {}
    num_filters = len(weight[0]) #CHANGED
    
    #compute the L1-norm of each filter weight and store it in a dictionary
    for j in range(num_filters):
        w_s = np.sum(abs(weight[j])) #CHANGED
        filt = 'filt_{}'.format(j)
        weights_dict[filt] = w_s
        
    #sort the filter as per their ascending L1 value
    weights_dict_sort = sorted(weights_dict.items(), key=lambda kv: kv[1])
    print('L1 norm conv layer {}\n'.format(i+1),weights_dict_sort)
    
    #get the L1-norm of weights from the dictionary and plot it
    weights_value = []
    for elem in weights_dict_sort:
        weights_value.append(elem[1])
        
    xc = range(num_filters)
    
    plt.figure(i+1,figsize=(7,5))
    plt.plot(xc,weights_value)
    plt.xlabel('Filter num')
    plt.ylabel('L1 norm')
    plt.title('conv layer {}'.format(i+1))
    plt.grid(True)
    plt.style.use(['classic'])


#%%
#Conv2D layers
#w1 = model.layers[0].layers[2].get_weights()[0]
#w2 = model.layers[0].layers[9].get_weights()[0]
#w3 = model.layers[0].layers[12].get_weights()[0]


w1 = model.layers[3].get_weights()

conv_layer_weights = w1

#for i in range(len(conv_layer_weights)):
    weight = conv_layer_weights[i]
    weights_dict = {}
    num_filters = len(weight[0]) #CHANGED
    
    #compute the L1-norm of each filter weight and store it in a dictionary
    for j in range(num_filters):
        w_s = np.sum(abs(weight[j])) #CHANGED
        filt = 'filt_{}'.format(j)
        weights_dict[filt] = w_s
        
    #sort the filter as per their ascending L1 value
    weights_dict_sort = sorted(weights_dict.items(), key=lambda kv: kv[1])
    print('L1 norm conv layer {}\n'.format(i+1),weights_dict_sort)
    
    #get the L1-norm of weights from the dictionary and plot it
    weights_value = []
    for elem in weights_dict_sort:
        weights_value.append(elem[1])
        
    xc = range(num_filters)
    
    plt.figure(i+1,figsize=(7,5))
    plt.plot(xc,weights_value)
    plt.xlabel('Filter num')
    plt.ylabel('L1 norm')
    plt.title('conv layer {}'.format(i+1))
    plt.grid(True)
    plt.style.use(['classic'])
    
#%%
layer_1 = model.layers[0].layers[423]
rmv = weights_dict_sort[0:int(0.25*len(weights_dict_sort))]
arr = np.zeros(len(rmv))
for k in range(len(rmv)):
    arr[k] = int(''.join(filter(str.isdigit, rmv[k][0])))
a = [int(i) for i in arr]
del arr
model_new = delete_channels(model.layers[0], layer_1, a)
del model
#%%

train_datagen = ImageDataGenerator(rescale = 1 / 255.0)
val_datagen = ImageDataGenerator(rescale = 1 / 255.0)
BS = 16
NUM_EPOCHS = 7
TRAIN_PATH = 'train'
VAL_PATH = 'valid'
TEST_PATH = 'test'
shape = [224,224]

trainGen = train_datagen.flow_from_directory(
			TRAIN_PATH,
			class_mode="categorical",
			target_size=(shape[0],shape[1]),
			color_mode="rgb",
			shuffle=True,
			batch_size=BS)
valGen = val_datagen.flow_from_directory(
			VAL_PATH,
			class_mode="categorical",
			target_size=(shape[0],shape[1]),
			color_mode="rgb",
			shuffle=False,
			batch_size=BS)
testGen = val_datagen.flow_from_directory(
			TEST_PATH,
			class_mode="categorical",
			target_size=(shape[0],shape[1]),
			color_mode="rgb",
			shuffle=False,
			batch_size=BS)
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

model = model_new

model.layers[1].trainable = True
'''set_trainable = False
print("\n\n\n"+str(conv_base.layers))'''
#count = 0
for layer in model.layers[1].layers:
		#if(count > 39):
			layer.trainable = True
		#count = count + 1
model.summary()
			
model.compile(loss = 'categorical_crossentropy',
	        	optimizer = optimizers.adam(lr=0.00001),
	        	metrics = ['acc'])
H = model.fit_generator(
			trainGen,
			steps_per_epoch=totalTrain // BS,
			epochs=NUM_EPOCHS)

model.save("Experimental4.h5")
print("Saved model to disk")

		

testGen.reset()
model = load_model("Experimental4.h5")
predIdxs = model.predict_generator(testGen,verbose=2,
						steps=(totalTest // BS) + 1)
		# for each image in the testing set we need to find the index of the
		# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)


		# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
			target_names=testGen.class_indices.keys()))

cm = confusion_matrix(testGen.classes, predIdxs)                               #CHANGED EVERY testGen to valGen
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1] + cm[2,2] + cm[3,3]) / total
#acc = (cm[0, 0] + cm[1, 1]) / total