# some basic functions
import os
import numpy as np
from PIL import Image
import json
import tensorflow as tf


def dataset(image_dir, label, image_hight = 128, image_width = 128):
	"""
	Get the resized images' piexl values from directory.
	Build a corresponding label array.

	Returns:
	X_orig, Y_orig -- arrays contains images' information and their labels
	"""
	X_orig, Y_orig = [], []
	for item in os.listdir(image_dir):
		image_orig = Image.open(image_dir + '\\' + item)
		image_resize = image_orig.resize((image_hight, image_width))
		image_array = np.array(image_resize)
		X_orig.append(image_array)
		Y_orig.append(label[item])
		
	X_orig = np.array(X_orig)
	Y_orig = np.array(Y_orig)
	Y_orig = Y_orig.reshape((1, Y_orig.shape[0]))

	return X_orig, Y_orig


def convert_json(json_file_path):
	"""
	Get the information from a json file.

	Returns:
	label -- a dictionary contains label_id through image_id
	"""
	with open(json_file_path, 'r', encoding = 'utf-8') as f:
		label_list = json.load(f)

	label = {}
	for item in label_list:
		label[item['image_id']] = int(item['label_id'])

	return label


def get_parameters():
	"""
	Initializes weight parameters to build a neural network with tensorflow.

	Returns:
	parameters -- a dictionary contains W1, W2, W3, W4
	"""
	#tf.set_random_seed(1)

	#conv-1
	W1_1 = tf.get_variable("W1_1", [3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
	W1_2 = tf.get_variable("W1_2", [3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
	#conv-2
	W2_1 = tf.get_variable("W2_1", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
	W2_2 = tf.get_variable("W2_2", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
	#conv-3
	W3_1 = tf.get_variable("W3_1", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
	W3_2 = tf.get_variable("W3_2", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
	W3_3 = tf.get_variable("W3_3", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
	#conv-4
	W4_1 = tf.get_variable("W4_1", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
	W4_2 = tf.get_variable("W4_2", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
	W4_3 = tf.get_variable("W4_3", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())

	parameters = {"W1_1": W1_1, "W1_2": W1_2,
				  "W2_1": W2_1, "W2_2": W2_2,
				  "W3_1": W3_1, "W3_2": W3_2, "W3_3": W3_3,
				  "W4_1": W4_1, "W4_2": W4_2, "W4_3": W4_3 }
	
	return parameters