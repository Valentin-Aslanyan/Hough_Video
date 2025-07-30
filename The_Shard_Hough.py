
"""
Realistic example of the Hough Line alogrithm with The Shard skyscraper in London
The grayscale image is first conditioned with erosion + dilation, then the Canny algorithm
The lines identified by the Hough transform are then plotted first by transforming them to the form
  y = m*x + c
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


#Read in image as grayscale,namely, one brightness value in the range [0,255]
img = cv2.imread('TheShard.png', cv2.IMREAD_GRAYSCALE)
height,width = np.shape(img)
assert img is not None, "file could not be read, check with os.path.exists()"


#Process the image: erode, then dilate to smooth the busy windows etc
erosion_size = 4
erosion_shape = cv2.MORPH_ELLIPSE
element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
img_eroded = cv2.erode(img, element)
img_dilated = cv2.dilate(img_eroded, element)


#Now the Canny algorithm to obtain edge pixels
img_canny = cv2.Canny(img_dilated, 50, 200, None, 3)


#Hough transform
Hough_lines = cv2.HoughLines(img_canny, 1, np.pi / 180, 200, None, 0, 0)


#On plot that runs between xlims, ylims, draw a line in the polar representation
def plot_Hough_line(r, theta, xlims, ylims):
	#Swap limits if out of order
	if xlims[0]>xlims[1]:
		xlims[1],xlims[0] = xlims[0],xlims[1]
	if ylims[0]>ylims[1]:
		ylims[1],ylims[0] = ylims[0],ylims[1]

	#Vertical line - see if it's inside plot and then draw it with constant x
	if np.sin(theta) == 0.0:
		x = r*np.cos(theta)
		if (x>=xlims[0]) and (x<=xlims[1]):
			plt.plot([x,x],[ylims[0],ylims[1]],color="red") 
	#Horizontal line - see if it's inside plot and then draw it with constant y
	elif np.cos(theta) == 0.0:
		y = r*np.sin(theta)
		if (y>=ylims[0]) and (y<=ylims[1]):
			plt.plot([xlims[0],xlims[1]],[y,y],color="red") 
	#Line with arbitrary gradient
	#First, convert it to the form y = m*x + c
	#Then, see where the line intercept each of the four lines around the plot (namely: above, below, left, right)
	else:
		m = -np.cos(theta)/np.sin(theta)
		c = r/np.sin(theta)

		x = []
		y = []

		y_intercept1 = m*xlims[0] + c
		if (y_intercept1>=ylims[0]) and (y_intercept1<=ylims[1]):
			x.append(xlims[0])
			y.append(y_intercept1)
		y_intercept2 = m*xlims[1] + c
		if (y_intercept2>=ylims[0]) and (y_intercept2<=ylims[1]):
			x.append(xlims[1])
			y.append(y_intercept2)
		x_intercept1 = (ylims[0] - c)/m
		if (x_intercept1>=xlims[0]) and (x_intercept1<=xlims[1]):
			x.append(x_intercept1)
			y.append(ylims[0])
		x_intercept2 = (ylims[1] - c)/m
		if (x_intercept2>=xlims[0]) and (x_intercept2<=xlims[1]):
			x.append(x_intercept2)
			y.append(ylims[1])
		if len(x)==2:
			plt.plot(x,y,color="red")


#Plot using Matplotlib
plt.figure(figsize=(17,8))
plt.subplot(1,4,1)
plt.imshow(img, cmap = 'gray')

plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,2)
plt.imshow(img_dilated, cmap = 'gray')

plt.title('Erosion + Dilation')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,3)
plt.imshow(img_canny, cmap = 'gray')

plt.title('Canny')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,4)
plt.imshow(img_canny, cmap = 'gray')
if Hough_lines is not None:
	for i in range(len(Hough_lines)):
		r = Hough_lines[i][0][0]
		theta = Hough_lines[i][0][1]
		plot_Hough_line(r, theta, [0,width], [0,height])

plt.xlim([0,width])
plt.ylim([height,0])
plt.title('Hough')
plt.xticks([])
plt.yticks([])

plt.show()



