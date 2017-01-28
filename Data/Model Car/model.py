# This file is used to generate model.json and model.h5

# TODO: Add regularizers to stop the car from zig zagging down the road

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
use_center_images_only = False
steering_angle = 0.25

# Hyper parameters
adam_learning_rate = 0.00001
samples_per_epoch = 60000
epoch_no = 2

# Modify image path
# Image path is with respect to full file path
# Use ./IMG
def modify_image_path(recorded_path, image_path):
    if image_path is None:
        return recorded_path
    else:
        return str(image_path) + "/" + str(os.path.split(recorded_path)[1])

def get_adjusted_steering_angle(steering_angle):
    if steering_angle < 0:
        return max(-1.0, steering_angle)
    else:
        return min(1.0, steering_angle)

def random_flipper(image, steering_angle, control = None):
    if control is None:
        control = choice([True, False])
    if control:
        image = cv2.flip(image, 1) # Flip horizontally
        steering_angle = -steering_angle
        return image, steering_angle
    else:
        return image, steering_angle

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

# class image_generator:
#     def __init__(self):
#         self.positive_steering_angle_proportion = 0.3
#         self.negative_steering_angle_proportion = 0.3
#         self.centered_steering_angle_proportion = 0.3
#         self.tolerence = 0.1
#         self.positive_images = 0
#         self.negative_images = 0
#         self.centered_images = 0
#         self.min_images = 1000
#
#     def print_stats(self):
#         f = open('stats.txt', 'w')
#         total_images = self.positive_images + self.negative_images
#         f.write("\nPositive Propostion: " + str(float(self.positive_images) / float(total_images)))
#         f.write("\nNegative Propostion: " + str(float(self.negative_images) / float(total_images)))
#         f.write("\nCenter Proposition: " + str(float(self.centered_images) / float(total_images)))
#         f.write("\nPositive Images: " + str(self.positive_images))
#         f.close()
#
#     def flip_decide(self, steering_angle):
#         # False - Don't need to flip
#         # True - please flip it
#         total_image = self.positive_images + self.negative_images + self.centered_images
#         if total_image == 0:
#             total_image = 1
#         negative_prop = self.negative_images/float(total_image)
#         positive_prop = self.positive_images/float(total_image)
#
#         print(negative_prop)
#         print(positive_prop)
#
#         if total_image < self.min_images:
#             return False
#
#         if steering_angle < 0.0:
#             if negative_prop < (self.negative_steering_angle_proportion + self.tolerence):
#                 return True
#             else:
#                 return False
#         else:
#             if positive_prop > (self.positive_steering_angle_proportion + self.tolerence):
#                 return True
#             else:
#                 return False
#
#     def register_angle(self, steering_angle):
#         if steering_angle < 0.0:
#             self.negative_images = self.negative_images + 1
#         elif steering_angle > 0.0:
#             self.positive_images = self.positive_images + 1
#         else:
#             self.centered_images = self.centered_images + 1
#

# Generator function
def generate_image(csv_path, steering_adj, center_images_only, image_path = None):
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

                #image_center = cv2.resize(image_center, (image_center.shape[0], image_center.shape[0]))
                #image_left = cv2.resize(image_left, (image_left.shape[0], image_left.shape[0]))
                #image_right = cv2.resize(image_right, (image_right.shape[0], image_right.shape[0]))

                yield np.array([image_center, image_left, image_right]), \
                      np.array([steering_angle_center, steering_angle_left, steering_angle_right])

        except Exception as e:
                    print(str(e))

    f.close()


# Getting image shape
image_shape = image1.shape
# image_shape = (image1.shape[0], image1.shape[0], 3)


model = Sequential()

model.add(Convolution2D(24, 5, 5, subsample=(2,2), input_shape=image_shape))
model.add(ELU())
model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Dropout(0.5))

model.add(Convolution2D(48, 3, 3, subsample=(2,2)))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, subsample=(2,2)))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, subsample=(2,2)))
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
model.compile(adam, "mse", ['accuracy'])

history = model.fit_generator(generate_image("driving_log.csv", steering_adj=steering_angle,
                                             center_images_only=use_center_images_only,
                                             image_path="./IMG"),
                              samples_per_epoch=samples_per_epoch, nb_epoch=epoch_no)

# model.json is the file that contains the model specifications
json_string = model.to_json()
f = open('model.json', 'w')
f.write(json_string)
f.close()

# model.h5 is the file that contains the weights of the model specified in model.json
model.save_weights("model.h5")
