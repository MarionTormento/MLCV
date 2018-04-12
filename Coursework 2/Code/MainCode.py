import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import signal
import time
import random

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
	plt.title("Ix")
	plt.subplot(232), plt.imshow(Iy, cmap='gray', interpolation='nearest')
	plt.title("Iy")
	plt.subplot(233), plt.imshow(Ixx, cmap='gray', interpolation='nearest')
	plt.title("Ixx")
	plt.subplot(234), plt.imshow(Iyy, cmap='gray', interpolation='nearest')
	plt.title("Iyy")
	plt.subplot(235), plt.imshow(Ixy, cmap='gray', interpolation='nearest')
	plt.title("Ixy")
	plt.suptitle('Intensity derivatives')


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
	thresholdCorner = np.percentile(R, 97)
	cornerPoints = np.where(R > thresholdCorner)
	NN = 10

	# Find local maxima for the corners
	maxCornerPointsX, maxCornerPointsY = local_maxima(R, cornerPoints, NN)
	CornerPoints = (np.asarray(maxCornerPointsX), np.asarray(maxCornerPointsY))

	thresholdEdge = np.percentile(R, 3)
	edgePoints = np.where(R < thresholdEdge)
	# Find local minima for the edges
	maxEdgePointsX, maxEdgePointsY = local_maxima(R, edgePoints, NN)
	EdgePoints = (np.asarray(maxEdgePointsX), np.asarray(maxEdgePointsY))
	
	# Plot
	plt.figure()
	plt.imshow(intensity, cmap='gray')
	plt.scatter(maxCornerPointsY, maxCornerPointsX, color='r', marker='+')
	plt.scatter(maxEdgePointsY, maxEdgePointsX, color='g', marker='+')
	plt.title("Detection of Corners and Edges")
	plt.show()

	return R, CornerPoints, EdgePoints

def local_maxima(R, Points, NN):

	# Function to compute the local maxima of the corners or edges by computing the max among the nearest neighbour of each corner point
	PointsX = Points[0]
	PointsY = Points[1]
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
				if abs(R[X[k]][Y[k]]) > Rmax:
					Rmax = abs(R[X[k]][Y[k]]) 
					Xmax = X[k]
					Ymax = Y[k]

		# Save the new corner index		
		localMaxPointsX.append(Xmax)
		localMaxPointsY.append(Ymax)

	return localMaxPointsX, localMaxPointsY

def descripter_funct(CornerPoints, OriginalImage):

	# Finds simple descriptors based on the intensity derivative in the square region around
	# interest points. From CW2 Q1.2: "...descriptor (simple colour and/or gradient orientation histogram)" -
	# so think this is the right thing to do.
	# We then have to "Implement a method that performs nearest neighbour matching of descriptors."
	# LUCKILY you are the queen of KNN :) :) So for one image we make a M-D space where M is the number of
	# Ix and Iy intensities we have in the region around the interest points. For the rest of the images we
	# then need to KNN to these clusters in the M-D space. I think that as this doesn't consider orientation,
	# zoom etc. it will suck. Which is why we then need to go on to do a SIFT descriptor (Q1.3). Thoughts? 

	# Set
	box = np.ones((5,5,3))
	# boxY = np.ones((4,4))
	img = cv2.imread('Photos/' + OriginalImage)
	print(img.shape)
	print(type(img.shape))

	for i in range(0,1): #len(CornerPoints[0])):
		print(CornerPoints[0][i]-2)
		print(CornerPoints[0][i]+3)
		print(CornerPoints[1][i]-2)
		print(CornerPoints[1][i]+3)

		box[:,:,:] = img[CornerPoints[0][i]-2:CornerPoints[0][i]+3, CornerPoints[1][i]-2:CornerPoints[1][i]+3,:]
		# boxY[:][:] = Iy[CornerPoints[0][i]-2:CornerPoints[0][i]+2][:,CornerPoints[1][i]-2:CornerPoints[1][i]+2]
		# box = np.concatenate((boxX,boxY))
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv2.calcHist([box[:,:,i]],[i],None,[256],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,256])
		plt.show()


def hog(Ix, Iy, CornerPoints):
	# Compute the magnitude of the gradient
	gradMagnitude = (Ix**2+Iy**2)**(1/2)
	
	# Compute the orientation of the gradient
	endX, endY = Ix.shape
	gradOrientation = np.zeros((endX,endY))
	for i in range(endX):
		for j in range(endY):
			if Ix[i][j] == 0 and Iy[i][j] != 0:
				gradOrientation[i][j] = np.pi/2
			elif Ix[i][j] == 0 and Iy[i][j] == 0:
				gradOrientation[i][j] = 0
			else:
				gradOrientation[i][j] = np.arctan(Iy[i][j]/Ix[i][j])
				if gradOrientation[i][j] < 0:
					gradOrientation[i][j] = gradOrientation[i][j] + np.pi #makes sure every angle is positive value (rad)

	# Plotting
	plt.figure()
	plt.subplot(121), plt.imshow(gradMagnitude, cmap='gray', interpolation='nearest')
	plt.title("Gradient Magnitude")
	plt.subplot(122), plt.imshow(gradOrientation, cmap='gray', interpolation='nearest')
	plt.title("Gradient orientation")
	plt.show()

	# Calculate Histogram of Gradients in 8×8 cells
	# https://www.learnopencv.com/histogram-of-oriented-gradients/
	
	# 0 - Clean the side corner points
	idx = np.where(CornerPoints[0] < 2)
	CornerPoints = np.delete(CornerPoints, idx[0], 1)
	idx = np.where(CornerPoints[0] >= endX-2)
	CornerPoints = np.delete(CornerPoints, idx[0], 1)
	idx = np.where(CornerPoints[1] < 2)
	CornerPoints = np.delete(CornerPoints, idx[0], 1)
	idx = np.where(CornerPoints[1] >= endY-2)
	CornerPoints = np.delete(CornerPoints, idx[0], 1)
	
	# 1 - Extract the 8x8 submatrix of magnitude and orientation
	histOrientGrad = np.zeros((len(CornerPoints[0]),9))
	for i in range(len(CornerPoints[0])):
		boxMagn = gradMagnitude[CornerPoints[0][i]-2:CornerPoints[0][i]+3][:,CornerPoints[1][i]-2:CornerPoints[1][i]+3]
		boxOrient = gradOrientation[CornerPoints[0][i]-2:CornerPoints[0][i]+3][:,CornerPoints[1][i]-2:CornerPoints[1][i]+3]
		# 2 - Compute the 9 bin histogram for the 8x8 submatrix (0: 0, 1:20, ..., 8:160)
		for j in range(5):
			for k in range(5):
				magn = boxMagn[j][k]
				orient = boxOrient[j][k]
				idxMin = int(orient/(rad(20)))
				idxSup = idxMin + 1
				if idxSup > 8:
					idxSup = idxSup-9
				percSup = (orient-idxMin*rad(20))/rad(20)
				percMin = 1-percSup
				histOrientGrad[i][idxMin] = percMin*magn
				histOrientGrad[i][idxSup] = percSup*magn
		del boxMagn
		del boxOrient


	# Plotting
	plotList = random.sample(range(len(histOrientGrad)), 9)
	plt.figure()
	for i in range(9):
		idx = 330 + i + 1
		plt.subplot(idx), plt.hist(histOrientGrad[plotList[i]])
		plt.title(plotList[i])
	plt.suptitle('Histogram of Gradient for 9 random 8x8 cells')
	plt.show()

def rad(degree):
	radian = degree*np.pi/180
	return radian

# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

for i in range(1):

	intensity, shift = getImageIntensity('dice.jpg')

	Ix, Iy = derivatives(intensity, shift)
	
	sigma = 1.6*shift
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)
	
	R, CornerPoints, EdgePoints = cornerness_funct(intensity, GIxx, GIyy, GIxy, 0.05)

	# descripter_funct(CornerPoints, 'chess.jpg')
	hog(Ix, Iy, CornerPoints)

