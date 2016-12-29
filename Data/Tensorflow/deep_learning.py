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

