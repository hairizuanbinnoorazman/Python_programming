# Steering with End to End Deep Learning

## Model Architecture and Training Strategy

### What kind of model infrastruture is implemented here?

- The neural network uses convolution layers with appropriate filter sizes.
- Layers exist to introduce nonlinearity into the model.
- The data is normalized in the model.
- Sufficient details of the characteristics and qualities of the architecture, such as the type of model used,
the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
- Discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

### Measures taken to reduce overfitting the model

- Train/validation/test splits have been used, and the
- Model uses dropout layers or other methods to reduce overfitting.

### Hyperparemeter selection

- Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

### How is the training data obtained and selected?

- Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).
- how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was
generated and examples of images from the dataset should be included.