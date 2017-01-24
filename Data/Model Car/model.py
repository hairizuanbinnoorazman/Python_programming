# This file is used to generate model.json and model.h5

# Import the Keras library
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D

# Import Numpy
import numpy as np

# Import image reading capability
import matplotlib.image as mpimg

# Testing purposes
image1_name = 'image1.jpg'
image2_name = 'image2.jpg'
image3_name = 'image3.jpg'
image1 = mpimg.imread(image1_name)
image2 = mpimg.imread(image2_name)
image3 = mpimg.imread(image3_name)
images = np.array([image1, image2, image3])

# Getting image shape
image_shape = image1.shape

steering_angle = np.array([0.0, 0.0, 0.0])

model = Sequential()

model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=image_shape))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.compile('adam', "mse", ['accuracy'])
history = model.fit(images, steering_angle, nb_epoch=3)

# model.json is the file that contains the model specifications
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# model.h5 is the file that contains the weights of the model specified in model.json
model.save_weights("model.h5")