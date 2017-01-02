"""
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
"""