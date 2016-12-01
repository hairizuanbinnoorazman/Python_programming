# Need to install both pillow and matplotlib as well as numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
# imread function reads an image into a numpy array - if its a jpg, it will read as dimensions X, Y as well as 3 (To represent R,G,B)
# If its png that has alpha value, it will read as X,Y,4 (4 represents R,G,B,A)
image = mpimg.imread('straight_road.jpg')
print('This image is: ',type(image),
         'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0] # Answer: 960 (length)
xsize = image.shape[1] # Answer: 540 (height)
# You cannot just color_select = image. It will assign the color_select to the same object as image
# If you did any changes, it will reflect to image variable as well
color_select = np.copy(image)

# Define our color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Use a "bitwise OR" to identify pixels below the threshold
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
# Thresholds is just a np array that says true or false
color_select[thresholds] = [0,0,0]

# Display the image
plt.imshow(color_select)
