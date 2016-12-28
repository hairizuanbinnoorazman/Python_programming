'''
Installing tensorflow:
pip install tensorflow **(Don't use the tensorflow-gpu; it causes problems)**

List of functions in tensorflow to know
tf.constant
tf.Variable
tf.Session()
tf.initialize_all_variables()

# Math
tf.add
tf.sub
tf.div
tf.mul
tf.matmul (Matrix Multiplication)

# Matrix Stuff
tf.zeros
tf.truncated_normal
tf.reduce_sum -> This is basically collapsing a array of numbers into a single number

# Neural Network related Stuff
tf.nn.softmax - Use to classify datasets
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
Consists of 3 steps
1. Producing logits via Logistic Regression
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

