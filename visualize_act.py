import numpy as np
import matplotlib as mp
#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import os
import cv2 as cv
# Custom Modules
from crater_cnn import Network as cnn

cwd = os.getcwd()

model = cnn(img_shape=(30, 30, 1))
conv_layer1 = model.add_convolutional_layer(5, 16)
conv_layer2 = model.add_convolutional_layer(5, 36)
flat_layer1 = model.add_flat_layer()
fc_layer1 = model.add_fc_layer(size=128, use_relu=True)
fc_layer2 = model.add_fc_layer(size=2, use_relu=False)
model.finish_setup()
# model.set_data(data)

model_path = os.path.join(cwd, 'model.ckpt')
model.restore(model_path)

path = './images/tiles'
img = cv.imread(path + '/tile3_25.pgm', 0)
#img = img.astype(float)
img = cv.normalize(img, img, 0, 1, cv.NORM_MINMAX)


model.getActivations(conv_layer1, img)
model.getActivations(conv_layer2, img)
model.getActivations(flat_layer1, img)
model.getActivations(fc_layer1, img)
model.getActivations(fc_layer2, img)

# cnn.plotNNFilter(units)
