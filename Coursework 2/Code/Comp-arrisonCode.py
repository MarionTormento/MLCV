import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import signal
import time
import random
from tempfile import TemporaryFile

def getImageIntensity(image):

	# Load an color image in grayscale
	img = cv2.imread('Photos/' + image, 0)
	img = np.asarray(img)

	# Autonomously find the size of the shift window adapted to the image
	width, height = img.shape
	shift_w = int(width/100)
	shift_h = int(height/100)
	shift = max(3, min(shift_h, shift_w)) # In case the image is smaller than 50x50, shift = 3

	return img, shift

def derivatives(intensity, shift, plot):

	# Function to calculate the change in intensity in the image in the 
	# x and y directions across 'shift' number of pixels.
	# 	INPUTS: Image intensity and the number of pixels to calculate the 
	# 			derivative across
	# 	OUTPUT: Derivative of the intensity in the x and y directions (Ix, Iy)
	# 	PLOT  : The square of the derivatives of the pixels intensity in 
	# 	the x and y direction (Ixx, Iyy) and the product of Ix and Iy (Ixy)
	
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
	if plot == 1:
		plt.figure()
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
		plt.show()

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

def cornerness_funct(intensity, GIxx, GIyy, GIxy, alpha, buff, plot):

	# Function to calculate the locations of corners and edges in the image
	
	# Calculate R
	R = (GIxx)*(GIyy) - GIxy**2 - alpha*(GIxx + GIyy)**2

	# Based on each pixels value of R, determine if it is a corner or an edge
	# or neither. 
	NN = 50
	perc = 97
	
	# Corners
	thresholdCorner = np.percentile(R, perc)
	cornerPoints = np.where(R > thresholdCorner)
	# Delete the corner located on the sides
	cornerPoints = cleanSides(intensity, cornerPoints, buff)
	# Find local maxima for the corners
	cornerPoints = local_maxima(R, cornerPoints, NN)
	
	# Edges
	thresholdEdge = np.percentile(R, 100-perc)
	edgePoints = np.where(R < thresholdEdge)
	# Delete the edges located on the sides
	edgePoints = cleanSides(intensity, edgePoints, buff)
	# Find local minima for the edges
	edgePoints = local_maxima(R, edgePoints, NN)

	# Plot
	if plot == 1:
		plt.figure()
		plt.imshow(intensity, cmap='gray')
		plt.scatter(cornerPoints[1], cornerPoints[0], color='r', marker='+')
		plt.scatter(edgePoints[1], edgePoints[0], color='g', marker='+')
		plt.title("Detection of Corners and Edges")

	return R, cornerPoints, edgePoints

def cleanSides(intensity, Points, buff):
	# Function to delete the interest points located on the sides of the image
	halfBuff = (buff-1)/2
	endX, endY = intensity.shape

	idx = np.where(Points[0] < halfBuff)
	Points = np.delete(Points, idx[0], 1)
	idx = np.where(Points[0] >= endX-halfBuff)
	Points = np.delete(Points, idx[0], 1)
	idx = np.where(Points[1] < halfBuff)
	Points = np.delete(Points, idx[0], 1)
	idx = np.where(Points[1] >= endY-halfBuff)
	Points = np.delete(Points, idx[0], 1)

	return Points

def local_maxima(R, Points, NN):

	# Function to compute the local maxima of the corners or edges by computing the max among the nearest neighbour of each corner point
	PointsX = Points[0]
	PointsY = Points[1]
	nbPoints = len(PointsX)

	localMaxPointsX = []
	localMaxPointsY = []

	for i in range(nbPoints):

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

		# Save the new corner index if it is not already in the list
		isX = np.where(localMaxPointsX == Xmax)
		isY = np.where(localMaxPointsY == Ymax)
		isAlreadyIn = np.where(isX[0] == isY[0])
		if len(isAlreadyIn[0]) == 0:
			localMaxPointsX.append(Xmax)
			localMaxPointsY.append(Ymax)

	localMaxPoints = (np.asarray(localMaxPointsX), np.asarray(localMaxPointsY))
	return localMaxPoints

FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

Quick = (['chess.jpg', 'chess.png', 'dice.jpg'])

allIntensity = []
allPoints = []
allHOG = []
test = Test_images
windowSize = 5

for i in range(0,1):

	print("New image")
	intensity, shift = getImageIntensity('Leaf.jpg')

	print("Computing Intensity derivatives")
	Ix, Iy = derivatives(intensity, shift, 0)
	sigma = 1.6*shift
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)

	print("Identifying corners and edges")
	R, CornerPoints, EdgePoints = cornerness_funct(intensity, GIxx, GIyy, GIxy, 0.05, windowSize, 1)

	# --------------------- Harris Corner Detector ----------------------
	
	filename = 'Leaf.jpg'
	img = cv2.imread('Photos/' + filename)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[0,0,255]

	cv2.imshow('Inbuilt Harris: Detection of Corners and Edges',img)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()

	#  --------------------- Shi-Tomasi Dectector --------------------------

	img1 = cv2.imread('Photos/' + filename)
	gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

	corners1 = cv2.goodFeaturesToTrack(gray1,25,0.01,10)
	corners1 = np.int0(corners1)

	for i in corners1:
	    x,y = i.ravel()
	    cv2.circle(img1,(x,y),3,255,-1)

	plt.figure()
	plt.imshow(img1)
	plt.title("Inbuilt Shi-Tomasi: Detection of Corners and Edges")
	 
	plt.show()
