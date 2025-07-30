
"""
Loads in a simple drawing of the letter "A" and performs a Hough transform to the parameter space of straight lines
Lines are characterized by theta, the angle to the vertical, and r the distance of closest approach to the origin
For a given point (pixel of interest) with coordinates x_p, y_p, all such straight lines must satisfy
r = x_p * cos(theta) + y_p * sin(theta)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


#Read in image as grayscale,namely, one brightness value in the range [0,255]
img = cv2.imread('Example1.png', cv2.IMREAD_GRAYSCALE)
height,width = np.shape(img)
assert img is not None, "file could not be read, check with os.path.exists()"


#Find x,y coordinates of purely white pixels (brightness == 255)
#points[1,:] contains x coords, points[0,:] contains y coords
points = np.stack(np.where(img == 255))
num_points = np.shape(points)[1]


#Grids of r, theta
r_max=int(np.ceil(np.sqrt(height**2+width**2)))+1
theta_plot=np.linspace(0,360,num=361)
r_plot=np.linspace(-r_max,r_max,num=2*r_max+1)
r_plot,theta_plot = np.meshgrid(r_plot,theta_plot)


#Transform the zeroth point as follows:
#Given the equation for r above, the following expression
# | r - x_p * cos(theta) - y_p * sin(theta) |
#must tend to 0 where the given equation is satisfied;
#this is calculated for all r, theta combinations and then clipped to 1 at most
#Subtracting this from 1 gives a value in the range [0,1] for display purposes
Hough_transform_single = 1.0-np.clip(np.abs(r_plot-points[1,0]*np.cos(theta_plot*np.pi/180.0)-points[0,0]*np.sin(theta_plot*np.pi/180.0)),0.0,1.0)

#Add on the contribution from all the other points
Hough_transform_full = np.copy(Hough_transform_single)
for idx_p in range(1,num_points):
	Hough_transform_full+=1.0-np.clip(np.abs(r_plot-points[1,idx_p]*np.cos(theta_plot*np.pi/180.0)-points[0,idx_p]*np.sin(theta_plot*np.pi/180.0)),0.0,1.0)


#Plot using Matplotlib
fig1=plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
ax1=fig1.gca()
plt.pcolormesh(theta_plot,r_plot,Hough_transform_single,cmap="gray",vmin=0,vmax=2)

plt.title('Hough Transform of Single Bright Pixel')
plt.tick_params(axis='both', which='major',labelsize=25,direction='in',bottom=True, top=True, left=True, right=True)
plt.ylabel("$r$",fontsize=26)
ax1.set_xticks([0,90,180,270,360])
plt.xlim([0,360])

plt.subplot(2,1,2)
ax2=fig1.gca()
plt.pcolormesh(theta_plot,r_plot,Hough_transform_full,cmap="gray",vmin=0,vmax=max(Hough_transform_full.flatten()))

plt.title('Hough Transform of All Bright Pixels')
plt.tick_params(axis='both', which='major',labelsize=25,direction='in',bottom=True, top=True, left=True, right=True)
plt.ylabel("$r$",fontsize=26)
plt.xlabel("$\\theta$ [$^{\\circ}$]",fontsize=26)
ax2.set_xticks([0,90,180,270,360])
plt.xlim([0,360])

plt.show()



