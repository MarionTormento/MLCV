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
		plt.show()

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

def descripter_funct(CornerPoints, OriginalImage, plot):

	# Finds simple descriptors based on the intensity derivative in the square region around
	# interest points. From CW2 Q1.2: "...descriptor (simple colour and/or gradient orientation histogram)" -
	# so think this is the right thing to do.
	# We then have to "Implement a method that performs nearest neighbour matching of descriptors."
	# LUCKILY you are the queen of KNN :) :) So for one image we make a M-D space where M is the number of
	# Ix and Iy intensities we have in the region around the interest points. For the rest of the images we
	# then need to KNN to these clusters in the M-D space. I think that as this doesn't consider orientation,
	# zoom etc. it will suck. Which is why we then need to go on to do a SIFT descriptor (Q1.3). Thoughts? 

	# Set
	boxSize = 11
	lengthA = (boxSize-1)//2
	lengthB = (boxSize+1)//2
	boxImg = np.ones((boxSize,boxSize))
	# boxY = np.ones((4,4))
	imgread = cv2.imread('Photos/' + OriginalImage)
	img = cv2.split(imgread)
	print(len(img))

	for i in range(80,81): #len(CornerPoints[0])):
		print(CornerPoints[0][i]-lengthA)
		print(CornerPoints[0][i]+lengthB)
		print(CornerPoints[1][i]-lengthA)
		print(CornerPoints[1][i]+lengthB)

		mask = np.zeros(imgread.shape[:2], np.uint8)
		mask[CornerPoints[0][i]-lengthA:CornerPoints[0][i]+lengthB,CornerPoints[1][i]-lengthA:CornerPoints[1][i]+lengthB] = 255
		color = ('b','g','r')
		if plot == 1:
			plt.figure()
			for j,col in zip(img, color):
				histr = cv2.calcHist([j],[0],mask,[256],[0,256])
				plt.plot(histr,color = col)
				plt.xlim([0,256])
			plt.show()

def hog(Ix, Iy, CornerPoints, plot):
	# Function to compute the histogram of gradient orientation of each interest points
	# INPUTS : Intensity derivatives, I
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
	if plot == 1:
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
				if idxSup > 8 and idxMin > 8:
					idxSup = idxSup-9
					idxMin = idxMin-9
				elif idxSup > 8:
					idxSup = idxSup - 9
				percSup = (orient-idxMin*rad(20))/rad(20)
				percMin = 1-percSup
				histOrientGrad[i][idxMin] = percMin*magn
				histOrientGrad[i][idxSup] = percSup*magn
		del boxMagn
		del boxOrient

	# Plotting
	if plot == 1:
		plotList = random.sample(range(len(histOrientGrad)), 9)
		plt.figure()
		for i in range(9):
			idx = 330 + i + 1
			plt.subplot(idx), plt.bar(range(9), histOrientGrad[plotList[i]])
			plt.xticks(range(9), ('0', '20', '40', '60', '80', '100', '120', '140', '160'))
			plt.title(plotList[i])
		plt.suptitle('Histogram of Gradient for 9 random 8x8 cells')
		plt.show()

	return histOrientGrad

def rad(degree):
	radian = degree*np.pi/180
	return radian

def knn(imgBase, imgTest, hogBase, hogTest, pointBase, pointTest, plot):
	# Function to compute the matching interest point between two images using the HOG as a descriptor
	# INPUTS: full hog of the base image and test image (we are trying to match test with base)
	# OUTPUTS: list of nearest neighbour : i-th line is the index of the closest neighbour in Base of the i-th interest point of Test
	
	#len(hogTest)
	indexNN = []
	for i in range(1):
		# Store the hog of the descriptor we want to compare
		hogDesc = hogTest[i]
		hogDesc = hogDesc*np.ones(hogBase.shape)
		distance = (hogBase-hogDesc)**2
		distance = (np.sum(distance, axis=0))**(1/2)
		print(distance)
		indexNN.append(np.where(distance == np.amin(distance)))
		print("indexNN")
		print(indexNN)
	if plot == 1:
		# plotList = random.sample(range(len(hogTest)), 10)
		plotList = 0
		plotTest = pointTest[plotList][:]
		print("plotTest")
		print(plotTest)
		plotBase = pointBase[indexNN[0][plotList]][:]
		print("plotBase")
		print(plotBase)
		plt.subplot(121), plt.imshow(imgBase, cmap='gray')
		plt.scatter(plotBase[1], plotBase[0], color='r', marker='+')
		plt.subplot(122), plt.imshow(imgTest, cmap='gray')
		plt.scatter(plotTest[1], plotTest[0], color='r', marker='+')


	return indexNN

# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

Quick = (['chess.jpg', 'chess.png', 'dice.jpg'])

allIntensity = []
allPoints = []
allHOG = []
test = Quick
windowSize = 5

for i in range(len(test)):
	print("New image")
	image = test[i]
	intensity, shift = getImageIntensity(image)
	# shift = 5
	print("Computing Intensity derivatives")
	Ix, Iy = derivatives(intensity, shift, 0)
	sigma = 1.6*shift
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)

	print("Identifying corners and edges")
	R, CornerPoints, EdgePoints = cornerness_funct(intensity, GIxx, GIyy, GIxy, 0.05, windowSize, 1)
	
	# print("Computing histogram of gradient orientation")
	# # descripter_funct(CornerPoints, image, 0)
	# allHOG.append(hog(Ix, Iy, CornerPoints, 0))

	# allIntensity.append(intensity)
	# allPoints.append(CornerPoints)

# Test : comparison of the two chessboards
# u = knn(allIntensity[0], allIntensity[1], allHOG[0], allHOG[1], allPoints[0], allPoints[1], 1)
# print(u)
# allHOG = np.array(allHOG)
# np.savetxt('hogQuick', allHOG)
