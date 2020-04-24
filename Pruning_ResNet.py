"""Prunes channels from Inception V3 fine tuned on a small flowers data set.
see setup instructions in inception_flowers_tune.py
inception_flowers_tune.py must be run first
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
from keras.layers import Convolution2D
import keras.backend as K
import pandas as pd
import numpy as np
import tensorflow as tf
import os,csv

# dimensions of our images.
shape = [224,224]

name = 'ResNet101'
os.chdir('C:/Users/WW/Desktop/Cancer')


output_dir = 'ResNet101/breakhis_binary/'
train_data_dir = 'dataset/breakhis_binary/train/'
validation_data_dir = 'dataset/breakhis_binary/valid/'
test_data_dir = 'dataset/breakhis_binary/test/'
tuned_weights_path = output_dir+'tuned_weights.h5'
epochs = 10
batch_size = 8
val_batch_size = 32
percent_pruning = 5
total_percent_pruning = 25
#%%

def iterative_prune_model():
	# build the inception v3 network
	'''base_model = inception_v3.InceptionV3(include_top=False,
										  weights='imagenet',
										  pooling='avg',
										  input_shape=(299, 299, 3))
	print('Model loaded.')

	top_output = Dense(5, activation='softmax')(base_model.output)

	# add the model on top of the convolutional base
	model = Model(base_model.inputs, top_output)
	del base_model
	model.load_weights(tuned_weights_path)
	# compile the model with a SGD/momentum optimizer
	# and a very slow learning rate.
	model.compile(loss='categorical_crossentropy',
				  optimizer=SGD(lr=1e-4, momentum=0.9),
				  metrics=['accuracy'])'''
	model = load_model(name+"/3_img_Best_model#"+str(100)+'_'+ name + ".h5")
	'''
    MY CONTRIBUTION
	model_temp.layers[0].layers.append(model_temp.layers[1])
	model_temp.layers[0].layers.append(model_temp.layers[2])
	model_temp.layers[0].layers.append(model_temp.layers[3])
	model_temp.layers[0].layers.append(model_temp.layers[4])
	
	model = model_temp.layers[0]
	del model_temp'''
	# Set up data generators
	train_datagen = ImageDataGenerator(rescale = 1 / 255.0)
	val_datagen = ImageDataGenerator(rescale = 1 / 255.0)
	trainGen = train_datagen.flow_from_directory(
				train_data_dir,
				class_mode="categorical",
				target_size=(shape[0],shape[1]),
				color_mode="rgb",
				shuffle=True,
				batch_size=batch_size)
	valGen = val_datagen.flow_from_directory(
				validation_data_dir,
				class_mode="categorical",
				target_size=(shape[0],shape[1]),
				color_mode="rgb",
				shuffle=False,
				batch_size=batch_size)
	testGen = val_datagen.flow_from_directory(
				test_data_dir,
				class_mode="categorical",
				target_size=(shape[0],shape[1]),
				color_mode="rgb",
				shuffle=False,
				batch_size=batch_size)
	totalTrain = len(list(paths.list_images(train_data_dir)))
	totalVal = len(list(paths.list_images(validation_data_dir)))
	totalTest = len(list(paths.list_images(test_data_dir)))
	
	train_steps = trainGen.n // trainGen.batch_size
	val_steps = valGen.n // valGen.batch_size
	test_steps = testGen.n // testGen.batch_size

	# Evaluate the model performance ON TEST before pruning
	'''loss = model.evaluate_generator(testGen,
									testGen.n //
									testGen.batch_size)
	print('original model test loss: ', loss[0], ', acc: ', loss[1])'''
    
	# Incrementally prune the network, retraining it each time
	percent_pruned = 0
	# If percent_pruned > 0, continue pruning from previous checkpoint
	if percent_pruned > 0:
		checkpoint_name = ('ResNet101_pruning_' + str(percent_pruned)
						   + 'percent')
		model = load_model(output_dir + checkpoint_name + '.h5')
        
	hubs = get_hubs_last_conv_name(model.layers[0].layers)
	special_conv = []
	for conv in hubs.values():
		special_conv += conv
	total_channels = get_total_channels(model)
	n_channels_delete = int(math.floor(percent_pruning / 100 * total_channels))
    
	while percent_pruned < total_percent_pruning:
		# Prune the model
		#Don't touch layers 86,88,89,92,93,94
		apoz_df = get_model_apoz(model, valGen, special_conv)
		percent_pruned += percent_pruning
		print('pruning up to ', str(percent_pruned),
			  '% of the original model weights')

		base_model = prune_model(model, apoz_df, n_channels_delete)
		
		model_temp = models.Sequential()
		model_temp.add(base_model)
		model_temp.add(model.layers[1])
		model_temp.add(model.layers[2])
		model_temp.add(model.layers[3])
		model_temp.add(model.layers[4])

		del model
		model = model_temp
		del model_temp
		# Clean up tensorflow session after pruning and re-load model
		checkpoint_name = ('ResNet101_pruning_' + str(percent_pruned)
						   + 'percent')
		model.save(output_dir + checkpoint_name + '.h5')
		print("Saved model to disk")
		del model
		K.clear_session()
		tf.reset_default_graph()
		model = load_model(output_dir + checkpoint_name + '.h5')

		# Re-train the model
		model.compile(loss = 'categorical_crossentropy',
				optimizer = optimizers.adam(lr=0.00001),
				metrics = ['acc'])
		checkpoint_name = ('ResNet101_pruning_' + str(percent_pruned)
						   + 'percent')
		csv_logger = CSVLogger(output_dir + checkpoint_name + '.csv')
		model.fit_generator(trainGen,
							steps_per_epoch=train_steps,
							epochs=epochs,
							validation_data=valGen,
							validation_steps=val_steps,
							workers=4,
							callbacks=[csv_logger])
	# Evaluate the final model performance ON TEST
		loss = model.evaluate_generator(testGen,
									testGen.n //
									testGen.batch_size)
		print('pruned model test loss: ', loss[0], ', acc: ', loss[1])
		model.save(output_dir+checkpoint_name+'_TRAINED.h5')
		with open(output_dir + checkpoint_name + '_TEST.csv', 'w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(["test loss", "test acc"])
				writer.writerow([loss[0], loss[1]])

def prune_model(model, apoz_df, n_channels_delete):
	# Identify 5% of channels with the highest APoZ in model
	sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
	high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

	# Create the Surgeon and add a 'delete_channels' job for each layer
	# whose channels are to be deleted.
	surgeon = Surgeon(model.layers[0], copy=True)  #EDITED model into model.layers[0]
	for name in high_apoz_index.index.unique().values:
		channels = [chan for chan in list(pd.Series(high_apoz_index.loc[name, 'index'],
dtype=np.int64).values) if chan < model.layers[0].get_layer(name).filters] 
		surgeon.add_job('delete_channels', model.layers[0].get_layer(name),
						channels=channels)
	# Delete channels
	return surgeon.operate()


def get_total_channels(model):
	#start = None   
	#end = None 
	channels = 0
	for layer in model.layers[0].layers:    #EDITED model into model.layers[0]
		if layer.__class__.__name__ == 'Conv2D':
		#if isinstance(layer, Convolution2D) and layer.name not in special_conv:
			channels += layer.filters
	return channels


def get_model_apoz(model, generator, special_conv):
	# Get APoZ
	#start = None    
	#end = None  
	#minus 27 for ResNet101
	apoz = []
	for k in range(len(model.layers[0].layers)):   #EDITED model into model.layers[0]
		#if model.layers[0].layers[k].__class__.__name__ == 'Conv2D':
		layer = model.layers[0].layers[k]
		if isinstance(layer, Convolution2D) and layer.name not in special_conv:
			print(model.layers[0].layers[k].name)
			apoz.extend([(model.layers[0].layers[k].name, i, value) for (i, value)
in enumerate(get_apoz(model.layers[0], model.layers[0].layers[k], generator))]) #EDITED model into model.layers[0]

	layer_name, index, apoz_value = zip(*apoz)
	apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
							'apoz': apoz_value})
	apoz_df = apoz_df.set_index('layer')
	return apoz_df

def get_hubs_last_conv_name(layerss):
    '''
    Get hubs (Merge, Concat) last convolutional layer name
    :param layers: model.layers
    :return: the dict of hub, key is hub name, value is its input
    '''
    hubs = {}
    for i, layer in enumerate(layerss):
        if i == 0:
            continue
        name = layer.name
        input_layers = get_input_layer_name(layer)
        if len(input_layers) > 1:
            input_conv_layers = []
            for input_layer in input_layers:
                input_conv_layer_name = get_last_conv_layer_name(input_layer)
                input_conv_layers.append(input_conv_layer_name)
            hubs[name] = input_conv_layers
    return hubs

def get_input_layer_name(layer):
    '''
    Get input of one layer
    :param layer: layer instance
    :return: input layers
    '''
    input_layers = None
    nodes = layer._inbound_nodes
    if len(nodes) == 1:
        node = nodes[0]
        input_layers = node.inbound_layers
    return input_layers

def get_last_conv_layer_name(layer):
    '''
    Get last convolution/merge/concat layer name
    :param layer: layer instance
    :return: the last convolutional layer name or Merge, Concat layer (which has many inputs)
    '''
    name = layer.name
    aim_layer = layer
    while name != '':
        input_layers = get_input_layer_name(aim_layer)
        if input_layers != None:
            if len(input_layers) == 1:
                if isinstance(input_layers[0], Convolution2D):
                    name = input_layers[0].name
                    break
                else:
                    aim_layer = input_layers[0]
                    name = aim_layer.name
            elif len(input_layers) > 1:
                name = aim_layer.name
                break
            else:
                name = ''
        else:
            name = ''
    return name

if __name__ == '__main__':
	iterative_prune_model()
