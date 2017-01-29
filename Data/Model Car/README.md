# Steering with End to End Deep Learning

## How is the training data obtained and selected?

- In order to prevent scenarios of garbage in garbage out, we would collect the data in batches. E.g. Record before
 starting the curve and right after the curve, stop the recording. At this point, we would save the batched data in
 another folder and prepare for the next batch of data.
- As mentioned within the NVIDIA paper that was mentioned within the lesson, we could utilize the left and right camera
images and treat them as though the car was positioned slightly left and slightly right. We would use a slight turn angle
several blogposts and discussions mentioned a slight angle of 0.2 to the steering wheel but this was tested on the
model. If we drove around at the center of the road and collected 10,000 images, we would have collected '30000' images that
can be used for training.
- Instead of using keyboard input, we would use a joystick input. (In my case, I utilized a PS4 controller connected to
my computer)
- The dataset I collected is combined with the dataset that someone provided in the slack channel (More data is always better?)
- If observed from the code, we would have the split the csv file only; there is no need to rearrange all the image files.
Moving files from one folder to another is a pretty expensive process so instead, we would choose the data to be used for
training via the csv file.
- For examples of some of the images being captured, refer to image1.jpg or image2.jpg or image3.jpg (They are used to
determine the size of the image and the sizes are plugged directly into the model. This would allow the model to be used
on other different sized images)


## What kind of model infrastruture is implemented here?

The following model infrastructure which is similar to NVIDIA's End to End Deep Learning Model

- Convolution Layer (5x5) with stride 2 and depth 24
- Convolution Layer (5x5) with stride 2 and depth 36
- Convolution Layer (3x3) with stride 2 and depth 48
- Convolution Layer (3x3) with stride 2 and depth 48
- Convolution Layer (3x3) with stride 2 and depth 48
- Flatten
- Fully Connected Layer (1000 nodes)
- Fully Connected Layer (100 nodes)
- Fully Connected Layer (50 nodes)
- Fully Connected Layer (10 nodes)
- Fully Connected Layer (1 node)

To sum it up, the model here contains 5 convolution layers followed by 5 fully connected layers. Activations within the
model are all ELU units. The model is designed such that you would not need to declare and modify the initial dimensions
of the input to the model.

To introduce non-linearity within the model, the Relu activations were used. These are one of the similar activation units
available and would be less complex than other models.

### Approach taken for deriving and designing a model architecture fit for solving the given problem.

- Initial design is a copy of a design that already works (NVIDIA's end to end solution to driving)
- Simpler activation units were used rather than the more complex ones that were mentioned in forums and slack channels
such as ELU activations etc
- Models are modified from the initial model - Add a single layer and testing it with the track one at a time
- As mentioned within the help guides and forums and slack channels for this project, the dataset has to be 30000-40000 range
at the minimum, hence the choice for the large amount

## Measures taken to reduce overfitting the model

- One of the ways to reduce overfitting of the model is the dropout layers within the first 2 layers of the model. The dropout
model will drop 50% of the connections randomly which would mean only stronger signals would be used to predict the steering
wheel angle
- Dataset has been split into training, validation and test datasets. Training would comprise of 70% of the data.
30% of the dataset will be the test dataset. The testing code is not provided in the model.py file.

## Hyperparemeter selection

- Adam optimizer is used. Instead of using the default learning rate, we would use the smaller learning rate of 0.00001.
Although a lower learning rate is better for driving the car more accurately, it would actually require more epochs of
data to train the dataset.

## Learnings

These are some of the observations I found while experimenting
Reflection section

- The main important thing before attempting to design and test various models is to get the infrastructure and processes
in place. When I first started the project, I did not have those processes and infrastructures in place which resulted in
slower iterative cycles for testing different models. This was overcome by first setting up the AWS GPU cluster with the
required environments preloaded with a large amount of image data. (I use a python2 environment at work so I can't really
run the simulations that way; I need to setup an environment for writing the code, one for testing the model on the app and
the AWS GPU cluster)
- I attempted to used a tanh activation in the last layer of the model and that caused the car to continuously drift to
the left while it attempt to drive around the track. When replaced with a Relu activation, the car instead remained in
the center of the road
- There are times when the car would swing back and forth during simulation in the 2nd/3rd lap. This is caused by
insufficient computation power in the machine I am using to test the simulation. (It can be observed that the calculations
actually paused for a moment - which led to the car moving too close to the edge before it swung back to the centre.)

