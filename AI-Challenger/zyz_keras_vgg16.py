import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math
from PIL import Image
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import *
from keras.metrics import top_k_categorical_accuracy
from keras.optimizers import Adam

#define some variables
train_image_dir = './scene_train_images/'
test_image_dir = './scene_test_images/'
image_height, image_width = 198, 198
num_classes = 80
num_channels = 3
learning_rate = 0.00001
dropout_rate = 0.4
lambd = 0.00001
num_epochs = 1
batch_size = 32

K.set_image_data_format('channels_last')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#K.set_learning_phase(1)

def load_data():
	X_train = np.load('X_train.npy')
	Y_train = np.load('Y_train.npy')
	X_test = np.load('X_test.npy')
	Y_test = np.load('Y_test.npy')

	#X_train = np.array([1, 2, 3])
	#Y_train = np.array([1])

	print('X_train.shape: ' + str(X_train.shape))
	print('Y_train.shape: ' + str(Y_train.shape))
	print('X_test.shape: ' + str(X_test.shape))
	print('Y_test.shape: ' + str(Y_test.shape))

	return X_train, Y_train, X_test, Y_test

def top_3_accuracy(y_true, y_pred):
	return top_k_categorical_accuracy(y_true, y_pred, k=3)

def vgg16_model(X_train, Y_train, X_test, Y_test, learning_rate, dropout_rate, lambd, num_epochs, batch_size):
	
	model_vgg16 = VGG16(include_top=False, 
						weights='imagenet', 
						input_shape=(image_height,image_width,3), 
						pooling='avg')
	for layer in model_vgg16.layers:
		layer.trainable = False
	model_vgg16.get_layer('block5_conv1').trainable = True
	model_vgg16.get_layer('block5_conv2').trainable = True
	model_vgg16.get_layer('block5_conv3').trainable = True

	#fully connected
	X = model_vgg16.output
	X = Dense(4096, activation = 'relu')(X)
	X = Dropout(dropout_rate)(X)
	X = ActivityRegularization(l1 = 0.0, l2 = lambd)(X)
	X = Dense(4096, activation = 'relu')(X)
	X = Dropout(dropout_rate)(X)
	X = ActivityRegularization(l1 = 0.0, l2 = lambd)(X)
	X = Dense(num_classes, activation = 'softmax')(X)
	model = Model(inputs = model_vgg16.inputs, outputs = X)
	
	model.load_weights('vgg16_model_64.h5')
	model.compile(optimizer = Adam(lr=learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy', top_3_accuracy])
	hist = model.fit(X_train, Y_train, epochs = num_epochs, batch_size = batch_size, validation_data=(X_test, Y_test))
	
	'''
	preds = model.evaluate(X_test, Y_test)
	print("Loss = " + str(preds[0]))
	print("Test Accuracy = " + str(preds[1]))

	
	model.compile(optimizer = Adam(lr=learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
	preds_2 = model.evaluate(X_test, Y_test)
	print("Loss = " + str(preds_2[0]))
	print("Test Accuracy = " + str(preds_2[1]))
	'''

	return model, hist

X_train, Y_train, X_test, Y_test = load_data()
model, hist = vgg16_model(X_train, Y_train, X_test, Y_test, learning_rate, dropout_rate, lambd, num_epochs, batch_size)
#print(hist.history)
#model.save_weights('vgg16_model_64.h5')