import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import signal
import time

def getImageIntensity(image):

	# Load an color image in grayscale
	img = cv2.imread('Photos/' + image, 0)
	
	width, height = img.shape
	shift_w = int(width/50)
	shift_h = int(height/50)
	shift = min(shift_h, shift_w)

	return img, shift

def derivatives(intensity, shift):

	# Function to calculate the change in intensity in the image in the 
	# x and y directions across 'shift' number of pixels.
	# 	INPUTS: Image intensity and the number of pixels to calculate the 
	# 			derivative across
	# 	OUTPUT: Derivative of the intensity in the x and y directions (Ix, Iy)
	# 	PLOT  : The square of the derivatives of the pixels intensity in 
	# 	the x and y direction (Ixx, Iyy) and the product of Ix and Iy (Ixy)
	
	# Method inspired from matlab solution on youtube
	# Find Ix and Iy
	intensity = np.asarray(intensity)

	# Find the derivatives in x and y direction
	half_shift = int(shift/2)
	u = np.linspace(-half_shift, half_shift, shift+1)
	dx, dy = np.meshgrid(u, u)

	Ix = signal.convolve2d(intensity, dx, 'same')
	Iy = signal.convolve2d(intensity, dy, 'same')


	# # Find Ix and Iy
	# intensity = np.asarray(intensity)
	# intensityShiftX = np.roll(intensity, shift, axis=1)
	# intensityShiftY = np.roll(intensity, shift, axis=0)

	# # Find the average change over 'shift' number of pixels
	# Ix = np.subtract(intensityShiftX, intensity) / shift
	# Iy = np.subtract(intensityShiftY, intensity) / shift

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

	# # Apply a gaussian window to the respective components of the 2x2 matrix
	# # containing the derivatives of the image intensity.
	# GIxx = ndimage.gaussian_filter(Ix**2, sigma)
	# GIyy = ndimage.gaussian_filter(Iy**2, sigma)
	# GIxy = ndimage.gaussian_filter(Ix*Iy, sigma)

	# return GIxx, GIyy, GIxy

	# Method inspired from matlab solution on youtube
	dim = max(1,6*sigma)
	m0 = -(dim-1)/2
	m = []
	n = []
	for i in range(0, dim):
		m.append(m0+i)
	for i in range(0,dim-1):
		n.append(m0+i)

	h1, h2 = np.meshgrid(m,n)
	hg = np.exp(-(h1 ** 2 + h2 ** 2)/(2*sigma^2))
	a, b = hg.shape
	sumtot = 0;
	for i in range(a):
		for j in range(b):
			sumtot = sumtot + hg[i,j]
	gauss_win = hg/sumtot

	GIxx = signal.convolve2d(Ix ** 2, gauss_win, 'same')
	GIyy = signal.convolve2d(Iy ** 2, gauss_win, 'same')
	GIxy = signal.convolve2d(Ix * Iy, gauss_win, 'same')

	return GIxx, GIyy, GIxy


def cornerness_funct(GIxx, GIyy, GIxy, alpha):

	# Function to calculate the locations of corners and edges in the image
	
	# Calculate R
	R = (GIxx)*(GIyy) - GIxy**2 - alpha*(GIxx + GIyy)**2
	# Rwidth, Rheight = R.shape
	# cornerPointsX = np.ones((1))
	# cornerPointsY = np.ones((1))

	# for i in range(1,Rheight-1):
	# 	for ii in range(1,Rwidth-1):
	# 		RBox = np.ones((3, 3))
	# 		for j in [0, 1, 2]:
	# 			for jj in [0, 1, 2]:
	# 				RBox[j][jj] = R[i-1+j][ii-1+jj]
	# 		maxR = np.where(RBox==np.amax(RBox))
	# 		# print(RBox)
	# 		print(maxR)
	# 		if (maxR[0] & maxR[1]) == 1:
	# 			print(i)
	# 			print(ii)
	# 			np.append(cornerPointsX, ii)
	# 			np.append(cornerPointsY, i)
	# 			# np.concatenate([cornerPointsX, ii], axis=0)
	# 			# np.concatenate([cornerPointsY, i], axis=0)
	# 		del RBox

	thresholdCorner = 0.5*np.amax(R)
	thresholdEdge = 0.5*np.amin(R)
	# Based on each pixels value of R, determine if it is a corner or an edge
	# or neither. 
	cornerPoints = np.where(R > thresholdCorner)
	cornerPointsX = cornerPoints[1]
	cornerPointsY = cornerPoints[0]
	edgePoints = np.where(R < thresholdEdge)
	edgePointsX = edgePoints[1]
	edgePointsY = edgePoints[0]

	# Plot
	plt.figure()
	plt.imshow(R, cmap='gray', interpolation='nearest')
	plt.scatter(cornerPointsX, cornerPointsY, color='r', marker='+')
	plt.scatter(edgePointsX, edgePointsY, color='g', marker='+')
	plt.show()

	return R

# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

for i in range(0,1):

	# intensity = getImageIntensity(HD[i])
	intensity, shift = getImageIntensity('chess.png')

	Ix, Iy = derivatives(intensity, shift)
	
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, 6, shift)
	
	R = cornerness_funct(GIxx, GIyy, GIxy, 0.01)
	print(np.amax(R))
	print(np.amin(R))
	