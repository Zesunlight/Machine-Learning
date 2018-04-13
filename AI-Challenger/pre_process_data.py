import numpy as np
import json
import os
from PIL import Image
from data_utils import *
from network_utils import *

#define some variables
train_image_dir = './scene_train/images/'
test_image_dir = './scene_test/images/'
image_height, image_width = 128, 128
num_classes = 80
num_channels = 3

#pre-treated train set
train_json_path = './scene_train_annotations.json'
train_label = convert_json(train_json_path)
X_train_orig, Y_train_orig = dataset(train_image_dir, train_label, image_height, image_width)
X_train = X_train_orig / 255.
numpy.save('X_train.npy', X_train)
num_train_examples = X_train.shape[0]
Y_train = np.eye(num_classes)[Y_train_orig.reshape(-1)]    #convert to one hot
numpy.save('Y_train.npy', Y_train)
print('pre-treated train set')

#pre-treated test set
test_json_path = './scene_test_annotations.json'
test_label = convert_json(test_json_path)  
X_test_orig, Y_test_orig = dataset(test_image_dir, test_label, image_height, image_width)
X_test = X_test_orig / 255.
np.save('X_test.npy', X_test)
num_test_examples = X_test.shape[0]
Y_test = np.eye(num_classes)[Y_test_orig.reshape(-1)]    #convert to one hot
np.save('Y_test.npy', Y_test)
print('pre-treated test set')
