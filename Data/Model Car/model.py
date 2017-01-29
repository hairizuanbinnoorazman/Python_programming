# This file is used to generate model.json and model.h5

# Import the Keras library
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import ELU

# Import Numpy
import numpy as np
import cv2

# Import image reading capability
import matplotlib.image as mpimg

# Import random to determine a random line of record to read
from random import randint
from random import random
from random import choice

import os
import csv

############################################
# Split the dataset
############################################

# # This part is disabled after writing the first version of the file
# from sklearn.cross_validation import train_test_split
#
# data = np.loadtxt('driving_log_full.csv', dtype=str, usecols=(0, 1, 2, 3))
# X_input_loc = data[:, (0, 1, 2)]
# y_output_angle = data[:, (3)]
# X_train, X_validation, y_train, y_validation = train_test_split(X_input_loc, y_output_angle, test_size=0.3, random_state=42)
#
# training_data = np.column_stack((X_train, X_validation))
# testing_data = np.column_stack((y_train, y_validation))
#
# training_data.tolist()
# testing_data.tolist()
#
# datafile = open('driving_log.csv', 'w')
# writer = csv.writer(datafile)
# writer.writerows(training_data)
#
# datafile_test = open('driving_log_test.csv', 'w')
# writer_test = csv.writer(datafile_test)
# writer_test.writerows(testing_data)

############################################
# Training the model
############################################

# Testing loading of images
image1_name = 'image1.jpg'
image2_name = 'image2.jpg'
image3_name = 'image3.jpg'
image1 = mpimg.imread(image1_name)
image2 = mpimg.imread(image2_name)
image3 = mpimg.imread(image3_name)
images = np.array([image1, image2, image3])

# Settings
csv_path = "driving_log.csv"
use_center_images_only = False
steering_angle = 0.25

# Hyper parameters
adam_learning_rate = 0.00001
samples_per_epoch = 30000
epoch_no = 10


def modify_image_path(recorded_path, image_path):
    """
    Get image path
    Returns the new image path although the data may say that the data is to be located on another computer/folder
    :param recorded_path: The path that is found in the dataset
    :param image_path: The folder where the images are currently found
    :return: Normalized image data
    """
    if image_path is None:
        return recorded_path
    else:
        return str(image_path) + "/" + str(os.path.split(recorded_path)[1])


def get_adjusted_steering_angle(steering_angle):
    """
    Get adjusted steering angles.
    It ensures that the steering angle will stay 1 or -1
    :param steering_angle: The steering angle that is to be processed
    :return: Normalized image data
    """
    if steering_angle < 0:
        return max(-1.0, steering_angle)
    else:
        return min(1.0, steering_angle)


def random_flipper(image, steering_angle, control = None):
    """
    Flips image and steering angle randomly
    :param image: The image data to be processed/flipped
    :param steering_angle: The steering angle that is to be processed or flipped
    :param control: Optional field. DEPRECIATED
    :return: Normalized image data
    """
    if control is None:
        control = choice([True, False])
    if control:
        image = cv2.flip(image, 1) # Flip horizontally
        steering_angle = -steering_angle
        return image, steering_angle
    else:
        return image, steering_angle


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


def generate_image(csv_path, steering_adj, center_images_only, image_path = None):
    """
    Image Generator function
    :param csv_path: The location of the image data
    :param steering_adj: The steering wheel adjustment if one is using the left/right camera images. Only activated when
    center_images_only is False
    :param center_images_only: Parameters to test
    :return: Normalized image data
    """

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

            if center_images_only:
                # Image, we would be doing a probability
                # We wouldn't want to use too much front driving - more concerned on curves
                # Mainly use the centre images - 50% chance
                if random() > 0.5 or center_images_only:
                    image = mpimg.imread(modify_image_path(str.strip(data[0]), image_path))
                elif random() > 0.5:
                    # Reads the right camera
                    image = mpimg.imread(modify_image_path(str.strip(data[2]), image_path))
                    steering_angle = min(1.0, steering_angle - steering_adj)
                else:
                    # Reads the left camera
                    image = mpimg.imread(modify_image_path(str.strip(data[1]), image_path))
                    steering_angle = min(1.0, steering_angle + steering_adj)

                # Further image manipulations here

                # Image normalization
                image = normalize(image)

                # TODO: Move image up or down to simulate slope
                # TODO: Darken or lighten the image
                # TODO: Cast shadow on image

                # Image resizing
                cv2.resize(image, (image.shape[0], image.shape[0]))

                yield np.array([image]), np.array([steering_angle])

            else:
                # Center image manipulation
                image_center = mpimg.imread(modify_image_path(str.strip(data[0]), image_path))
                steering_angle_center = steering_angle
                image_center, steering_angle_center = random_flipper(image_center, steering_angle_center)

                # Left image manipulation
                image_left = mpimg.imread(modify_image_path(str.strip(data[1]), image_path))
                steering_angle_left = steering_angle + steering_adj
                image_left, steering_angle_left = random_flipper(image_left, steering_angle_left)

                # Right image manipulation
                image_right = mpimg.imread(modify_image_path(str.strip(data[2]), image_path))
                steering_angle_right = steering_angle - steering_adj
                image_right, steering_angle_right = random_flipper(image_right, steering_angle_right)

                yield np.array([image_center, image_left, image_right]), \
                      np.array([steering_angle_center, steering_angle_left, steering_angle_right])

        except Exception as e:
                    print(str(e))

    f.close()


# Getting image shape
image_shape = image1.shape

############################################
# Defining Model
############################################

model = Sequential()

model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=image_shape))
model.add(ELU())
model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, subsample=(2,2), W_regularizer=l2(0.01)))
model.add(ELU())
model.add(Dropout(0.5))

model.add(Convolution2D(48, 3, 3, subsample=(2,2), W_regularizer=l2(0.01)))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, subsample=(2,2), W_regularizer=l2(0.01)))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, subsample=(2,2), W_regularizer=l2(0.01)))
model.add(ELU())

model.add(Flatten())

model.add(Dense(1000, W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
model.add(ELU())

model.add(Dense(500, W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
model.add(ELU())

model.add(Dense(100, W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
model.add(ELU())

model.add(Dense(50, W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
model.add(ELU())

model.add(Dense(10, W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
model.add(ELU())
# model.add(Activation('relu'))
model.add(Dense(1))

adam = Adam(lr=adam_learning_rate)
#model.compile(adam, "mse", ['accuracy'])
model.compile(adam, "mse")

############################################
# Run function to train model
############################################

history = model.fit_generator(generate_image("driving_log.csv", steering_adj=steering_angle,
                                             center_images_only=use_center_images_only,
                                             image_path="./IMG"),
                              samples_per_epoch=samples_per_epoch, nb_epoch=epoch_no)

############################################
# Write the output
############################################

# model.json is the file that contains the model specifications
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# model.h5 is the file that contains the weights of the model specified in model.json
model.save_weights("model.h5")
