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

region_select = np.copy(image)
# Define a triangle region of interest
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz
left_bottom = [0, 539]
right_bottom = [959, 539]
apex = [480, 300]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# Create a best straight line?
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
# Most of the pixels would be false?
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

region_select = np.copy(image)
# Color pixels red which are inside the region of interest
region_select[~(thresholds | ~region_thresholds)] = [255, 0, 0]

# Display the image
plt.imshow(region_select)
