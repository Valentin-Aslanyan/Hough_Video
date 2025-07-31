# A quick Hough Transform explanation

See the video here: [https://www.youtube.com/watch?v=caNUo-bQV9c](https://www.youtube.com/watch?v=caNUo-bQV9c)

![a](https://img.youtube.com/vi/caNUo-bQV9c/0.jpg "Hough Transform: Algorithms for Grad Students (2)")

The Hough Transform is a technique in applied mathematics to allow curves with a particular equation (think of straight lines, parabolas, circles...) to be found among groups of pixels.

You will need `Python` with `numpy`, `matplotlib` and (optionally) `cv2` which you should install using `pip` or whatever other package manager you have.

`Hough_Line.py` and `Hough_Circle.py` will transform to the parameter space of lines/circles from whence you could identify "knots". For the hand-drawn letter A and the (:|) emoji, the coordinates of the bright pixels are identified. You will see a plot of a transformation to the space of:

- r,θ (polar representation) for lines 
- x_C,y_C (center coordinates) for circles

A full Hough Transform algorithm would then choose any point in this parameter space above a certain threshold (a knot).

`The_Shard_Hough.py` is a practical example of identifying lines in a real image, of 32 London Bridge Street, London in this case. The color image is first conditioned by:

- Loading it in grayscale
- Applying erosion and dilation 
- Applying the Canny algorithm

Then, the Hough Line Transform is used to identify any long straight lines.
