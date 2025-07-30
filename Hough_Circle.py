
"""
Loads in a simple drawing of an emoji "(:|)" and performs a Hough transform to the parameter space of circles
All circles have a radius of target_r pixels from a center characterized by coordinates x_c, y_c
For a given point (pixel of interest) with coordinates x_p, y_p, all such circles must satisfy
r^2 = (x_c-x_p)^2 + (y_c-y_p)^2
"""

#Desired size of circle in pixels
target_r = 40 


import numpy as np
import matplotlib.pyplot as plt
import cv2


#Read in image as grayscale,namely, one brightness value in the range [0,255]
img = cv2.imread('Example2.png', cv2.IMREAD_GRAYSCALE)
height,width = np.shape(img)
assert img is not None, "file could not be read, check with os.path.exists()"


#Find x,y coordinates of purely white pixels (brightness == 255)
#points[1,:] contains x coords, points[0,:] contains y coords
points = np.stack(np.where(img == 255))
num_points = np.shape(points)[1]


#Grids of x_c, y_c
x_c_plot=np.linspace(-target_r,width+target_r,num=width+2*target_r+1)
y_c_plot=np.linspace(-target_r,height+target_r,num=height+2*target_r+1)
x_c_plot,y_c_plot = np.meshgrid(x_c_plot,y_c_plot)


#Transform the zeroth point as follows:
#Given the equation for r above, the following expression
# | r - sqrt( (x_c-x_p)^2 + (y_c-y_p)^2 ) |
#must tend to 0 where the given equation is satisfied;
#this is calculated for all x_c, y_c combinations and then clipped to 1 at most
#Subtracting this from 1 gives a value in the range [0,1] for display purposes
Hough_transform_single = 1.0-np.clip(np.abs(target_r-np.sqrt((x_c_plot-points[1,0])**2+(y_c_plot-points[0,0])**2)),0.0,1.0)

#Add on the contribution from all the other points
Hough_transform_full = np.copy(Hough_transform_single)
for idx_p in range(1,num_points):
	Hough_transform_full+=1.0-np.clip(np.abs(target_r-np.sqrt((x_c_plot-points[1,idx_p])**2+(y_c_plot-points[0,idx_p])**2)),0.0,1.0)


#Plot using Matplotlib
fig1=plt.figure(figsize=(5,8))
plt.subplot(2,1,1)
ax1=fig1.gca()
plt.pcolormesh(x_c_plot,y_c_plot,Hough_transform_single,cmap="gray",vmin=0,vmax=2)

plt.title('Hough Transform of Single Bright Pixel')
plt.tick_params(axis='both', which='major',labelsize=18,direction='in',bottom=True, top=True, left=True, right=True)
plt.ylabel("$y_c$",fontsize=26)
plt.ylim([np.shape(img)[0]+target_r,-target_r])

plt.subplot(2,1,2)
ax2=fig1.gca()
plt.pcolormesh(x_c_plot,y_c_plot,Hough_transform_full,cmap="gray",vmin=0,vmax=max(Hough_transform_full.flatten()))

plt.title('Hough Transform of All Bright Pixels')
plt.tick_params(axis='both', which='major',labelsize=18,direction='in',bottom=True, top=True, left=True, right=True)
plt.ylabel("$y_c$",fontsize=26)
plt.xlabel("$x_c$",fontsize=26)
plt.ylim([np.shape(img)[0]+target_r,-target_r])

plt.show()



