# Steering with End to End Deep Learning

### How is the training data obtained and selected?

- In order to prevent scenarios of garbage in garbage out, we would collect the data in batches. E.g. Record before
 starting the curve and right after the curve, stop the recording. At this point, we would save the batched data in
 another folder and prepare for the next batch of data.
- As mentioned within the NVIDIA paper that was mentioned within the lesson, we could utilize the left and right camera
images and treat them as though the car was positioned slightly left and slightly right. We would use a slight turn angle
several blogposts and discussions mentioned a slight angle of 0.25 to the steering wheel but this was tested on the
model. If we drove around at the center of the road and collected 10,000 images, we would have collected '30000' images that
can be used for training.
- In order to understand the dataset further, after the initial processing of utilizing both the left and right hand camera images,
we would vizualize the distribution of steering angles to see where it is most concentrated at. This is done via graphs

### What kind of model infrastruture is implemented here?

- The neural network uses convolution layers with appropriate filter sizes.
- Layers exist to introduce nonlinearity into the model.
- The data is normalized in the model.
- Sufficient details of the characteristics and qualities of the architecture, such as the type of model used,
the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
- Discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

### Measures taken to reduce overfitting the model

- One of the ways to reduce overfitting of the model is the dropout layers within the first 2 layers of the model. The dropout
model will drop 50% of the connections randomly which would mean only stronger signals would be used to predict the steering
wheel angle
- To reduce overfitting further, we would do extra adjustments to the images such as changing the increasing and reducing
the brightness of the images
- Train/validation/test splits have been used, and the


### Hyperparemeter selection

- Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

