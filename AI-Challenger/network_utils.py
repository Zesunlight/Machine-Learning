#modules relavent to my network
import tensorflow as tf
import numpy as np
import math


def forward_propagation(X, parameters, keep_prob):
	#conv - 1
	#128*128*3 -> 128*128*32 -> 128*128*32
	W1_1 = parameters['W1_1']
	Z1_1 = tf.nn.conv2d(X, W1_1, strides=[1,1,1,1], padding='SAME')
	#A1_1 = tf.nn.relu(Z1_1)
	
	W1_2 = parameters['W1_2']
	Z1_2 = tf.nn.conv2d(Z1_1, W1_2, strides=[1,1,1,1], padding='SAME')
	#A1_2 = tf.nn.relu(Z1_2)
	
	#max pool - 1
	#128*128*32 -> 64*64*32
	P1 = tf.nn.max_pool(Z1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
	
	#conv - 2
	#64*64*32 -> 64*64*64 -> 64*64*64
	W2_1 = parameters['W2_1']
	Z2_1 = tf.nn.conv2d(P1, W2_1, strides=[1,1,1,1], padding='SAME')
	#A2_1 = tf.nn.relu(Z2_1)
	
	W2_2 = parameters['W2_2']
	Z2_2 = tf.nn.conv2d(Z2_1, W2_2, strides=[1,1,1,1], padding='SAME')
	#A2_2 = tf.nn.relu(Z2_2)
	
	#max pool - 2
	#64*64*64 -> 32*32*64
	P2 = tf.nn.max_pool(Z2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
	
	#conv - 3
	#32*32*64 -> 32*32*128 ~ (3)
	W3_1 = parameters['W3_1']
	Z3_1 = tf.nn.conv2d(P2, W3_1, strides=[1,1,1,1], padding='SAME')
	#A3_1 = tf.nn.relu(Z3_1)
	
	W3_2 = parameters['W3_2']
	Z3_2 = tf.nn.conv2d(Z3_1, W3_2, strides=[1,1,1,1], padding='SAME')
	#A3_2 = tf.nn.relu(Z3_2)
	
	W3_3 = parameters['W3_3']
	Z3_3 = tf.nn.conv2d(Z3_2, W3_3, strides=[1,1,1,1], padding='SAME')
	#A3_3 = tf.nn.relu(Z3_3)
	
	#max pool - 3
	#32*32*128 -> 16*16*128
	P3 = tf.nn.max_pool(Z3_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
	
	#conv - 4
	#16*16*128 -> 16*16*128 ~ (3)
	W4_1 = parameters['W4_1']
	Z4_1 = tf.nn.conv2d(P3, W4_1, strides=[1,1,1,1], padding='SAME')
	#A4_1 = tf.nn.relu(Z4_1)
	
	W4_2 = parameters['W4_2']
	Z4_2 = tf.nn.conv2d(Z4_1, W4_2, strides=[1,1,1,1], padding='SAME')
	#A4_2 = tf.nn.relu(Z4_2)
	
	W4_3 = parameters['W4_3']
	Z4_3 = tf.nn.conv2d(Z4_2, W4_3, strides=[1,1,1,1], padding='SAME')
	#A4_3 = tf.nn.relu(Z4_3)
	
	#max pool - 4
	#16*16*128 -> 8*8*128
	P4 = tf.nn.max_pool(Z4_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
	
	#flatten
	F = tf.contrib.layers.flatten(P4)
	
	#fully connected
	Z5 = tf.contrib.layers.fully_connected(F, 1024)
	A5 = tf.nn.dropout(Z5, keep_prob)
	
	Z6 = tf.contrib.layers.fully_connected(A5, 256)
	A6 = tf.nn.dropout(Z6, keep_prob)
	
	Z7 = tf.contrib.layers.fully_connected(A6, 80, activation_fn=None)
	
	return Z7


def random_minibatches(X, Y, mini_batch_size = 64, seed = 0):
	"""
	Creates a list of random minibatches from (X, Y)
	
	Arguments:
	X -- input data, of shape (m, Hi, Wi, Ci)
	Y -- true "label" vector of shape (m, n_y)
	mini_batch_size - size of the mini-batches, integer
	seed -- keep the random values steady, easily for tuning
	
	Returns:
	mini_batches -- list of (mini_batch_X, mini_batch_Y)
	
	"""
	m = X.shape[0]                  # number of training examples
	mini_batches = []
	#np.random.seed(seed)
	
	# Shuffle (X, Y) change their position
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:,:,:]
	shuffled_Y = Y[permutation,:]

	# Partition (shuffled_X, shuffled_Y). Minus the end case.
	# number of mini batches of size mini_batch_size in partitionning
	num_complete_minibatches = math.floor(m/mini_batch_size) 
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches