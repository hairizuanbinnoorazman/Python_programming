'''
Installing tensorflow:
pip install tensorflow **(Don't use the tensorflow-gpu; it causes problems)**

List of functions in tensorflow to know
tf.constant
tf.Variable
tf.placeholder
tf.Session()
tf.initialize_all_variables() # Depreciated by 2017
tf.train.Saver() as saver; saver.save(); saver.restore(tf.Session(), <save file>)
tf.reset_default_graph() # Resets bias and weights

# Math
tf.add
tf.sub
tf.div
tf.mul
tf.matmul (Matrix Multiplication)
tf.random_normal

# Matrix Stuff
tf.zeros
tf.truncated_normal
tf.reduce_sum -> This is basically collapsing a array of numbers into a single number via its sum
tf.reduce_mean -> This is basically collapsing an array of numbers into a single number via its mean

# Neural Network related Stuff
tf.nn.softmax - Use to classify datasets
tf.train.GradientDescentOptimizer - Using Gradient Descent as numerical
'''

import tensorflow as tf

############################################################
# Hello World in Tensorflow
############################################################

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)

############################################################
# Inputs in Tensorflow
############################################################

x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    print output

############################################################
# Simple Math in Tensorflow
############################################################

# Doing simple math with tensorflow
x = tf.constant(10)
y = tf.constant(2)
z = tf.sub(tf.div(x,y),tf.constant(1)) # If you do without tf.constant(1), it will cause errors

with tf.Session() as sess:
    output = sess.run(z)
    print(output)

############################################################
# Simple Math in Tensorflow
############################################################

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    output = sess.run(init)
    weights = weights
    print(sess.run(tf.zeros(5)))


############################################################
# Softmax in Tensorflow
############################################################

def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)

    softmax = tf.nn.softmax(logits)

    with tf.Session() as sess:
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output

############################################################
# Multinomial Logistic Classification in Tensorflow
############################################################

'''
Some definitions:
parameters are things like weights and biases.
hyper-parameters and things like training rates etc

Consists of 3 steps
1. Producing logits via Logistic Regression (Matrix Multiplication of weights and input and add bias unit)
2. Logits (Probabilities) are further tuned to their respective classifications with softmax
3. Value after softmax is compared via cross-entropy function to one hot encoding (Belong to one class, the rest are zero)

We can then use cross entropy to calculate the loss which we can then use to optimize.
However, calculating the loss is too costly on compute and data access.
(Essentially, you are calculating the direction on where to go with the dataset by calculating it across the ten of thousands of rows of data)

Important things to note - to make optimizer's life easier, need to try to go for the mean and variance of the dataset
input - mean and variance of dataset (Normalize dataset)
weights - random, centered around mean and have a small variance

How, there is method to calculate this via SGD (Stochastic Gradient Descent - do gradient descent on a sample of the dataset)
When doing the learning, we can utilize the momentum of the gradient descent (running average) to reduce noise of the gradient descent
and reduce learning rate as we get closer and closer to the "answer"

Some hyper-parameters to take note:
- Initial Learning Rate
- Learning Rate Decay
- Momentum
- Batch Size
- Weight Initiailization
'''

############################################################
# Cross Entropy comparison
############################################################

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(tf.mul(one_hot, tf.log(softmax)))

with tf.Session() as sess:
    print(sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data}))

############################################################
# Full example with batching
############################################################

# One thing to remember is all tensorflow related stuff must use other tensorflow related stuff
# If you need a variable, make sure you use tensorflow variable

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import math

# Why batching? Not all computers are powerful.
# One way to deal with batching is to have algorithms that learn "slowly"
# Keep feeding it data. Big datasets take a long time to crunch.
# Smaller datasets allows you to iteratively run your models over and over
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)

    return outout_batches

learning_rate = 0.001
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input]) # First entry is type, then shape, then name
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
# The first paramter is shape. You can classify no of rows and no of columns.
# (You have to surround it with square brackets)
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Remember what you would need to do:
# 1. Run the linear regression over the data to get logits
# 2. Get the logits and softmax them (Techniques involves trying to an exponential fraction - make strongers signal stronger)
# 3. Compare softmax-ed values with one hot to get cross-entropy
# 4. Sum up errors and feed that to gradient descent optimizer

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
# Tensorflow has a weird nature... 0 means row but it does operation column-wise, 1 means col
# This next statement is comparison the whole list of tensor to each other
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
# Technically, to calculate accuracy is total of correct/total but we can do that simply with this operation
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 128

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # TODO: Train optimizer on all batches
    for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))


"""
Configuring the following:

- epoch:
Higher epoch values allow user to train over and over again. It's like practise makes perfect principle

- batch:
Initially, you can say that bigger batch capture the most pictures. However, this is not the case. A smaller batch allows you to catch
the smaller details which cannot be captured in big batches
So, go smaller for batches


- training_rate:
A lower training rate is better. Training rate is the amount of changes accepted between current value and supposed correct value.
If too big, it could lead to huge swings in the training which would lead to ineffective training.
Lower training rate allow user to reach higher training rate. But too low, and it will overfit of sorts?
Must be somewhere in the middle of sorts
"""