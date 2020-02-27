# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:37:21 2020

@author: dias.azhigulov
"""
import numpy as np
import os
from imutils import paths
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


i = 1 #MALIGNANT DATASET TESTS
NUM_EPOCHS = 100
def func1(shape,name):
	BS = 16
	#os.chdir(r"D:/Capstone_results/ResNet50/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func2(shape,name):
	BS = 32
	#os.chdir(r"D:/Capstone_results/ResNet101/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func3(shape,name):
	BS = 16
	#os.chdir(r"D:/Capstone_results/ResNet152/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func4(shape,name):
	BS = 16
	#os.chdir(r"D:/Capstone_results/DenseNet121/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func5(shape,name):
	BS = 8
	#os.chdir(r"D:/Capstone_results/DenseNet201/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func6(shape,name):
	BS = 16
	#os.chdir(r"D:/Capstone_results/Xception/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func7(shape,name):
	BS = 32
	#os.chdir(r"D:/Capstone_results/InceptionV3/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func8(shape,name):
	BS = 16
	#os.chdir(r"D:/Capstone_results/VGG16/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

	return BS,conv_base

def func9(shape,name):
	BS = 16
	#os.chdir(r"D:/Capstone_results/VGG19/")
	conv_base = load_model("2_img_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")
	return BS,conv_base

#os.chdir(r"D:/Capstone_results/")
TEST_PATH = 'test/'
totalTest = len(list(paths.list_images(TEST_PATH)))

prediction = np.zeros(totalTest)
#model_names = ['InceptionV3','Xception','ResNet50','ResNet101','ResNet152',
 #            'DenseNet121','DenseNet201','VGG16','VGG19']
model_names = ['Xception']
predIdxs = np.zeros([totalTest,len(model_names)])
j = 0


for name in model_names:
	if ((name =='InceptionV3') or (name =='Xception')):
		shape = [299,299]
	else:
		shape = [224,224]

	if (name == 'ResNet50'): BS,conv_base = func1(shape,name)
	elif (name == 'ResNet101'): BS,conv_base = func2(shape,name)
	elif (name == 'ResNet152'): BS,conv_base = func3(shape,name)
	elif (name == 'DenseNet121'): BS,conv_base = func4(shape,name)
	elif (name == 'DenseNet201'): BS,conv_base = func5(shape,name)
	elif (name == 'Xception'): BS,conv_base = func6(shape,name)
	elif (name == 'InceptionV3'): BS,conv_base = func7(shape,name)
	elif (name == 'VGG16'): BS,conv_base = func8(shape,name)
	elif (name == 'VGG19'): BS,conv_base = func9(shape,name)
	
	val_datagen = ImageDataGenerator(rescale = 1 / 255.0)
	testGen = val_datagen.flow_from_directory(
            TEST_PATH,
			class_mode="categorical",
			target_size=(shape[0],shape[1]),
			color_mode="rgb",
			shuffle=False,
			batch_size=BS)

	testGen.reset()
	temp = conv_base.predict_generator(testGen,verbose=2,
						steps=(totalTest // BS) + 1)
	predIdxs[:,j] = np.argmax(temp, axis=1)
	j += 1
	del conv_base

#%%

predIdxs = predIdxs.astype('int64')

class0 = len(list(paths.list_images(TEST_PATH+'adenosis')))
class1 = len(list(paths.list_images(TEST_PATH+'fibroadenoma')))
class2 = len(list(paths.list_images(TEST_PATH+'phyllodes_tumor')))
class3 = len(list(paths.list_images(TEST_PATH+'tubular_adenoma')))

arr = np.zeros(totalTest)
arr[0:class0] = 0
arr[class0:class1] = 1
arr[class1:class2] = 2
arr[class2:class3] = 3

for k in range(totalTest):
    counts = np.bincount(predIdxs[k,:])
    prediction[k] = np.argmax(counts)
    
cm = confusion_matrix(testGen.classes,prediction)
total = sum(sum(cm))
acc = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/total
sensitivity0 = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0, 2] + cm[0, 3])
sensitivity1 = cm[1, 1] / (cm[1, 1] + cm[1, 0] + cm[1, 2] + cm[1, 3])
sensitivity2 = cm[2, 2] / (cm[2, 2] + cm[2, 0] + cm[2,1] + cm[2,3])
sensitivity3 = cm[3, 3] / (cm[3, 3] + cm[3, 0] + cm[3,1] + cm[3,2])
sensitivity = (sensitivity3+sensitivity2+sensitivity1+sensitivity0)/4

#fname1 = "noaug_modelHistory#"+str(NUM_EPOCHS)+'_'+ str(name) + ".txt"
fname1 = "Ensemble_9models.txt"
f = open(fname1,"w+")
f.write("\nSensitivity (avg): " + str(round(sensitivity,4)) + "; Test acc: " + str(round(acc,4)))
f.write("\nSensitivity0: {:.4f}".format(sensitivity0))
f.write("\nSensitivity1: {:.4f}".format(sensitivity1))
f.write("\nSensitivity2: {:.4f}".format(sensitivity2))
f.write("\nSensitivity3: {:.4f}".format(sensitivity3))
f.write("\n\n" + classification_report(testGen.classes, prediction,
target_names=testGen.class_indices.keys()))
f.write("\n" + str(cm))
f.close()