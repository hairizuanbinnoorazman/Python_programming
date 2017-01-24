# This file is used to generate model.json and model.h5

# Import the Keras library
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))



# model.json is the file that contains the model specifications
# model.h5 is the file that contains the weights of the model specified in model.json
