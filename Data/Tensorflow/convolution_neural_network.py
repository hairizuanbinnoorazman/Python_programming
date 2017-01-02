"""
Q. Why the need for Convolution Neural Networks?
If one is to train on image pixels directly, it would literally melt current computers. So, instead, we would convert them,
to detect "edges", "lines" and other features. That is why after CNN conversion, the "depth" of the image is not between 3 or 1
(3 usually refer to color, 1 is grayscale)

Q. What's the difference between VALID and SAME padding in the CNN algorithm?
VALID padding - refer to the algorithm trying to squeeze the filter along side the image.
SAME padding - refer to the algorithm adding additional pixels along the side such that after when it goes through the algo, it will
be able to use every once of the pixels without wasting a pixel.
VALID means its ok to lose some data and SAME means nope, must use every ounce of it.

Q. What is this depth parameter?
Technically, the depth should be higher. The higher the better. But too big will kill the system.

Q. How do you get the weights that is to be shared across a "row" of the output?
Mathematical computation - Need to find out but the tf.conv.2d operation seems to take care of this

Q. Reducing computation requirements. How to do that?
Convolutions can reduce spatial dimension size by using stride of 2 but you lose a lot of data along the way.
There are methods that deal with it such as pooling, 1 by 1 convolutions and inception.
MAX pooling - "Sharper images"
AVERAGE pooling = "Blur" images

Q. What is full connected layer?
Is just your regular Neural Network stuff - Linear network with a Relu unit to produce your logits

Q. What is Le-net infrastructure?
LeNet is one of the famous convolution neural network that allow users to train on data pretty accurately.
Consists of 2 convolutions with 3 fully connected networks
Each convolution is a convolution step, activation and pooling
Each fully connected network is a linear regression step followed by activation

Additional notes when implementing CNNs as tensorflow is below:
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    #
    # Weights are shared for a filter in a convolution.
    # Shape of weights: <filter_height>, <filter_width>, <current_depth>, <future_depth>
    #
    # Biases are also shared.
    # Shape of biases: <future_depth>
    #
    # Strides Shape (Determines how the filters move across the unit)
    # strides = <batch_skip_no> <height_skip_no> <width_skip_no> <depth_skip_no>
    # We wouldn't ever want to skip batches nor do we want to skip depths of data.
    # That's why most of the time, strides will usually be [1, X, Y, 1] where batch_skip_no and depth_skip_no is 1 (which means it doesn't skip)
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    #
    # Definition for stride is similar as above.
    #
    # Pooling
    # Refers to the window which is being looked at during that moment. So, similar to above...
    # ksize should be [1, X, Y, 1]. And in most cases, we want the following values:
    # ksize = [1, 2, 2, 1]
    # strides = [1, 2, 2, 1]
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 10), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits