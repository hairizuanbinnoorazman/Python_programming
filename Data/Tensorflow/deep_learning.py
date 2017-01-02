"""
What is deep learning?

Deep learning is an extension to what was done in the basic tensorflow.

Basic tensorflow -> Set up of a single Neural Network
An extension to this is to connect the basic output of that neural network to a hidden layer.
Instead of using the linear algebra again, use something non-linear such as ReLu unit

Why is deep learning effective?

Most data strucutres out there are actually hierachical in nature;
Take a picture of a face;
A computer would recognize lines and edges as part of the first layer
Build up to facial features such as mouth or eyebrows etc in the second or third layer
Final layer would be the face

Some major important to note:
- It is extremely easy for Deep Learning Models to overfit the data. As time goes time, it fits to the training data better
- However, overfitting is bad, if new examples come in, cannot classify it.
- Early Termination (Keep training and testing against validation until it drops - thats where you stop it)
- Regularization (Add a penalty term to prevent model complexity and huge parameters)
- Dropout (Don't allow model to rely on non-linear activation units (25-50% of the time, values dropped to zero)
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

n_hidden_layer = 256 # layer number of features

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1]) # First is type, second is shape. Having none shape means can take any number
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_input])

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

