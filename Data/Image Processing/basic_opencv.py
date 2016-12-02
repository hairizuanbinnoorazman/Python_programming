import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread('exit_ramp.png')
plt.imshow(image)

# Here we read a .png and convert to 0,255 bytescale
# png has alpha so its not so compatible immediately...
image = (mpimg.imread('exit_ramp.png')*255).astype('uint8')

import cv2  #bringing in OpenCV libraries
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion
plt.imshow(gray, cmap='gray')

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)


# Define parameters for Canny and run it
# NOTE: if you try running this code you might want to change these!
low_threshold = 100
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')