import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import signal
import time

def getImageIntensity(image):

	# Load an color image in grayscale
	img = cv2.imread('Photos/' + image, 0)
	
	# Autonomously find the size of the shift window adapted to the image
	width, height = img.shape
	shift_w = int(width/50)
	shift_h = int(height/50)
	shift = max(1, min(shift_h, shift_w)) # In case the image is smaller than 50x50, shift = 1

	return img, shift

def derivatives(intensity, shift):

	# Function to calculate the change in intensity in the image in the 
	# x and y directions across 'shift' number of pixels.
	# 	INPUTS: Image intensity and the number of pixels to calculate the 
	# 			derivative across
	# 	OUTPUT: Derivative of the intensity in the x and y directions (Ix, Iy)
	# 	PLOT  : The square of the derivatives of the pixels intensity in 
	# 	the x and y direction (Ixx, Iyy) and the product of Ix and Iy (Ixy)
	
	# Find Ix and Iy
	intensity = np.asarray(intensity)

	# Find the derivatives in x and y direction
	half_shift = int(shift/2)
	u = np.linspace(-half_shift, half_shift, shift+1)
	dx, dy = np.meshgrid(u, u)

	Ix = signal.convolve2d(intensity, dx, 'same')
	Iy = signal.convolve2d(intensity, dy, 'same')

	# Find the squares of these
	Ixx = Ix ** 2
	Iyy = Iy ** 2
	Ixy = Ix * Iy
	
	# Plotting
	figure = plt.figure()
	plt.subplot(231), plt.imshow(Ix, cmap='gray', interpolation='nearest')
	plt.subplot(232), plt.imshow(Iy, cmap='gray', interpolation='nearest')
	plt.subplot(233), plt.imshow(Ixx, cmap='gray', interpolation='nearest')
	plt.subplot(234), plt.imshow(Iyy, cmap='gray', interpolation='nearest')
	plt.subplot(235), plt.imshow(Ixy, cmap='gray', interpolation='nearest')

	del intensity

	return Ix, Iy

def gaussian_window(Ix, Iy, sigma, shift):

	# INPUTS: Derivatives of image intensity in the x and y directions
	# OUTPUTS: Gaussian filtered intensity derivatives.

	# Define the gaussian window sized by the shift
	m0 = -(shift-1)/2
	m = []
	n = []
	for i in range(0, shift):
		m.append(m0+i)
	for i in range(0,shift-1):
		n.append(m0+i)
	h1, h2 = np.meshgrid(m,n)
	gauss = np.exp(-(h1 ** 2 + h2 ** 2)/(2*sigma**2))
	sumtot = np.sum(gauss)
	gauss = gauss/sumtot

	# Convolution of the Intensity matrix and the gaussian window
	GIxx = signal.convolve2d(Ix ** 2, gauss, 'same')
	GIyy = signal.convolve2d(Iy ** 2, gauss, 'same')
	GIxy = signal.convolve2d(Ix * Iy, gauss, 'same')

	return GIxx, GIyy, GIxy

def cornerness_funct(image, GIxx, GIyy, GIxy, alpha):

	# Function to calculate the locations of corners and edges in the image
	
	# Calculate R
	R = (GIxx)*(GIyy) - GIxy**2 - alpha*(GIxx + GIyy)**2
	
	# Based on each pixels value of R, determine if it is a corner or an edge
	# or neither. 
	thresholdCorner = np.percentile(R, 95)
	cornerPoints = np.where(R > thresholdCorner)
	# Find local maxima for the corners
	maxCornerPointsX, maxCornerPointsY = local_maxima(R, cornerPoints, 8)

	thresholdEdge = np.percentile(R, 5)
	edgePoints = np.where(R < thresholdEdge)
	# Find local minima for the edges
	maxEdgePointsX, maxEdgePointsY = local_maxima(R, edgePoints, 8)
	
	# Plot
	plt.figure()
	plt.imshow(intensity, cmap='gray')
	# plt.imshow(R, cmap='gray', interpolation='nearest')
	plt.scatter(maxCornerPointsX, maxCornerPointsY, color='r', marker='+')
	plt.scatter(maxEdgePointsX, maxEdgePointsY, color='g', marker='+')
	plt.show()

	return R

def local_maxima(R, Points, NN):

	# Function to compute the local maxima of the corners or edges by computing the max among the nearest neighbour of each corner point
	PointsX = Points[1]
	PointsY = Points[0]
	nbPoints = len(PointsX)

	localMaxPointsX = []
	localMaxPointsY = []

	for i in range(0, nbPoints):

		Point0X = PointsX[i]*np.ones(nbPoints)
		Point0Y = PointsY[i]*np.ones(nbPoints)

		# Compute the distance between each corner point and cornerPoint0
		distanceX = (PointsX - Point0X)**2
		distanceY = (PointsY - Point0Y)**2
		distance = (distanceX + distanceY)**(1/2)
		distance = np.delete(distance, i, 0)

		# Looking for the maxima among the cornerPoint0 NN nearest neighbour
		Rmax = 0
		Xmax = 0
		Ymax = 0

		for j in range(NN):
		
			index = np.where(distance == np.amin(distance))
			distance[index[0]] = 100000
			Y = PointsY[index[0]]
			X = PointsX[index[0]]
			for k in range(len(Y)):
				if abs(R[Y[k]][X[k]]) > Rmax:
					Rmax = abs(R[Y[k]][X[k]]) 
					Xmax = X[k]
					Ymax = Y[k]

		# Save the new corner index		
		localMaxPointsX.append(Xmax)
		localMaxPointsY.append(Ymax)

	return localMaxPointsX, localMaxPointsY

# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

for i in range(len(Test_images)):

	intensity, shift = getImageIntensity(Test_images[i])

	Ix, Iy = derivatives(intensity, shift)
	
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, 1, shift)
	
	R = cornerness_funct(intensity, GIxx, GIyy, GIxy, 0.05)

