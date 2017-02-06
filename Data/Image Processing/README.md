# Image Processing

Contains a list of scripts on Image Processing using Python

Some of the things that one could do to images/video stills:

- Usual first steps would be to calibrate the camera. Allows you to remove the distortion caused by lens
- Change the color representation of the image. Sometimes, the task that you are doing may not require you to work on the RGB space. You can potentially work on the HLS or HSV color space.
- Perspective Transform. All images would contain some sort of perspective where further away object would look smaller and nearer objects would look larger. In the case of analyzing road images, you would want to understand the curvature of the road but this perspective stuff kind of makes it hard to do so. That is where you can do a perspective transform
- Perform Cannes or Sobel image transforms that try to get the gradient of the color space of the image and try to get more information on it

# List of Useful tutorials

1. Camera Calibration. https://github.com/hairizuanbinnoorazman/CarND-Camera-Calibration
