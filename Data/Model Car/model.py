# This file is used to generate model.json and model.h5

# Import the Keras library
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

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

# Settings
csv_path = "driving_log.csv"
use_center_images_only = True
steering_angle = 0.10

# Hyper parameters
adam_learning_rate = 0.00001
samples_per_epoch = 100
epoch_no = 2

# Min-Max Scaling
def normalize(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    min = 0
    max = 255
    return a + ( ( (image_data - min)*(b - a) )/( max - min ) )

# Generator function
def generate_image(csv_path, steering_adj, center_images_only):
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
            if random() > 0.5 or center_images_only:
                image = mpimg.imread(str.strip(data[1]))
                # print "Using " + data[1] + " steering_angle: " + str(steering_angle)
            elif random() > 0.5:
                # Reads the right camera
                image = mpimg.imread(str.strip(data[2]))
                steering_angle = min(1.0, steering_angle - steering_adj)
                # print "Using " + data[1] + " steering_angle: " + str(steering_angle)
            else:
                # Reads the right camera
                image = mpimg.imread(str.strip(data[0]))
                steering_angle = min(1.0, steering_angle + steering_adj)
                # print "Using " + data[1] + " steering_angle: " + str(steering_angle)

            # Further image manipulations here

            # Image normalization
            image = normalize(image)

            # TODO: Move image up or down to simulate slope
            # TODO: Darken or lighten the image
            # TODO: Cast shadow on image
            yield np.array([image]), np.array([steering_angle])

        except Exception as e:
            print str(e)

    f.close()


# Getting image shape
image_shape = image1.shape

model = Sequential()

model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=image_shape))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(128, 3, 3, subsample=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='tanh'))

adam = Adam(lr=adam_learning_rate)
model.compile(adam, "mse", ['accuracy'])
history = model.fit_generator(generate_image("driving_log.csv", steering_adj=steering_angle, center_images_only=use_center_images_only),
                              samples_per_epoch=samples_per_epoch, nb_epoch=epoch_no)

# model.json is the file that contains the model specifications
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# model.h5 is the file that contains the weights of the model specified in model.json
model.save_weights("model.h5")