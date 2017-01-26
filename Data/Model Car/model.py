# This file is used to generate model.json and model.h5

# Import the Keras library
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D

# Import Numpy
import numpy as np

# Import image reading capability
import matplotlib.image as mpimg

# Import random to determine a random line of record to read
from random import randint
from random import random

# Testing purposes
image1_name = 'image1.jpg'
image2_name = 'image2.jpg'
image3_name = 'image3.jpg'
image1 = mpimg.imread(image1_name)
image2 = mpimg.imread(image2_name)
image3 = mpimg.imread(image3_name)
images = np.array([image1, image2, image3])

# Generator function
def generate_image(csv_path, steering_adj = 0.15):
    # Get the file size
    f = open(csv_path)
    master_data = f.readlines()
    no_of_records = len(master_data)

    while True:
        try:
            # Generate a random line to be read
            line_no = randint(0, no_of_records)

            # Read line of data
            data = master_data[line_no-1]

            # Meanings of numerical fields in data
            # Steering, throttle, brake, speed
            # We would only be using steering - column 4 (0 indexed)
            data = data.split(",")
            steering_angle = float(data[3])

            # Image, we would be doing a probability
            # We wouldn't want to use too much front driving - more concerned on curves
            # Mainly use the centre images - 50% chance
            if random() > 0.5:
                image = mpimg.imread(str.strip(data[1]))
                print "Using " + data[1] + " steering_angle: " + str(steering_angle)
            elif random() > 0.5:
                # Reads the right camera
                image = mpimg.imread(str.strip(data[2]))
                steering_angle = min(1.0, steering_angle - steering_adj)
                print "Using " + data[1] + " steering_angle: " + str(steering_angle)
            else:
                # Reads the right camera
                image = mpimg.imread(str.strip(data[0]))
                steering_angle = min(1.0, steering_angle + steering_adj)
                print "Using " + data[1] + " steering_angle: " + str(steering_angle)

            # Further image manipulations here
            # TODO: Move image up or down to simulate slope
            # TODO: Darken or lighten the image
            # TODO: Cast shadow on image
            yield np.array([image]), np.array([steering_angle])

        except Exception as e:
            print str(e)

    f.close()


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
#history = model.fit(images, steering_angle, nb_epoch=3)
history = model.fit_generator(generate_image("driving_log.csv"), samples_per_epoch=100, nb_epoch=4)

# model.json is the file that contains the model specifications
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# model.h5 is the file that contains the weights of the model specified in model.json
model.save_weights("model.h5")