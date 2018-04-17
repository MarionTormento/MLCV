import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import signal
import time
import random
# from tempfile import TemporaryFile

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
	# Function to apply the gaussian window to each shift x shift frame of the image
	# INPUTS: Derivatives of image intensity in the x and y directions
	# OUTPUTS: Gaussian filtered intensity derivatives.

	# Define the gaussian window of size shift x shift
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

def cornerness_funct(intensity, GIxx, GIyy, GIxy, shift, alpha, buff, plot):

	# Function to calculate the locations of corners (and edges) in the image
	# Both Harris corner detection and Tomasi and Shi methods are implemented
	# INPUTS: Gaussian filtered intensities, alpha value for HCD, buffer to delete the sides interest points
	# OUTPUTS: Interest points coordinates
	# PLOT: Location of computed interest points on the image

	## Compute R
	# Harris Corner detection method
	R = (GIxx)*(GIyy) - GIxy**2 - alpha*(GIxx + GIyy)**2
	# Tomasi and Shi method
	# R = np.zeros(GIxx.shape)
	# endX, endY = GIxx.shape
	# for i in range(endX):
	# 	for j in range(endY):
	# 		M = np.array([[GIxx[i][j], GIxy[i][j]],[GIxy[i][j], GIyy[i][j]]])
	# 		eigVals = np.linalg.eigh(M)
	# 		R[i][j]  = np.amin(eigVals[0])

	# Based on each pixels value of R, determine if it is a corner or an edge
	# or neither. 
	NN = 50 # Number of Nearest Neighbour
	perc = 99 # Percentage of value kept by the thresholding
	halfShift = int(shift/2)
	## Corners
	# Threshold
	thresholdCorner = np.percentile(R, perc)
	cornerPoints = np.where(R > thresholdCorner)
	# Delete the corner located on the sides
	cornerPoints = cleanSides(intensity, cornerPoints, buff)
	# Find local maxima for the corners
	cornerPoints = local_maxima(R, cornerPoints, NN)

	# ## Edges
	# thresholdEdge = np.percentile(R, 100-perc)
	# edgePoints = np.where(R < thresholdEdge)
	# # Delete the edges located on the sides
	# edgePoints = cleanSides(intensity, edgePoints, buff)
	# # Find local minima for the edges
	# edgePoints = local_maxima(R, edgePoints, NN)

	# Plot
	if plot == 1:
		plt.figure()
		plt.imshow(intensity, cmap='gray')
		plt.scatter(cornerPoints[1], cornerPoints[0], color='r', marker='+')
		# plt.scatter(edgePoints[1], edgePoints[0], color='g', marker='+')
		plt.title("Detection of Corners and Edges")
		plt.show()

	return cornerPoints

def cleanSides(img, Points, buff):
	# Function to delete the interest points located on the sides of the image
	# INPUTS: Image to extract the shape and locate the sides, Points coordinates to be cleaned, size of the buffer window
	# OUTPUT: Points coordinates with the points located on the buff sides deleted
	
	# Compute the size of the buffer window and image
	halfBuff = int((buff-1)/2)
	endX, endY = img.shape

	# Deletion of points located on the sides
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
	# INPUTS : R value, coordinates of the Interest Points, and number of Nearest Neighbour
	# OUTPUT : Local maxima Interest Points coordinates

	# Initialisation of the variable
	PointsX = Points[0]
	PointsY = Points[1]
	nbPoints = len(PointsX)

	localMaxPointsX = []
	localMaxPointsY = []

	for i in range(nbPoints):
		# Save the coordinates of the points for which we are currently looking for its nearest neighbour
		Point0X = PointsX[i]
		Point0Y = PointsY[i]
		# Compute the distance between each point and Point0
		distance = np.linalg.norm([PointsX-Point0X,PointsY-Point0Y], axis=0)
		# Set its own distance value to max distance so that it is not taken for a nearest neighbour
		distanceMax = np.amax(distance)
		distance[i] = distanceMax

		# Looking for the maxima R among the cornerPoint NN nearest neighbour 
		Rmax = R[Point0X][Point0Y]
		Xmax = Point0X 
		Ymax = Point0Y

		for j in range(NN):
			# Looking for the index of the nearest neigbour (= minimal distance to Point0)
			index = np.where(distance == np.amin(distance))
			# Set its distance to max distance so it is not taken twice for a neighbour
			distance[index[0]] = distanceMax
			# Saves its coordinates
			Y = PointsY[index[0]]
			X = PointsX[index[0]]
			for k in range(len(Y)): # Security in case several points are equally distanced to Point0
				# If R of the neighbour is greater than the previous R, save the location of the point and R
				if abs(R[X[k]][Y[k]]) > Rmax: 
					Rmax = abs(R[X[k]][Y[k]]) 
					Xmax = X[k]
					Ymax = Y[k]

		# Save the new local maxima point if it is not already in the list
		isX = np.where(localMaxPointsX == Xmax) # indices of the local maxima with same X value
		isY = np.where(localMaxPointsY == Ymax) # indices of the local maxima with same Y value
		isAlreadyIn = np.where(isX[0] == isY[0]) # matching indices between isX and isY
		# if isX and isY indices matches somewhere, then it means the local maxima coordinates are already in the list
		if len(isAlreadyIn[0]) == 0:
			localMaxPointsX.append(Xmax)
			localMaxPointsY.append(Ymax)

	localMaxPoints = (np.asarray(localMaxPointsX), np.asarray(localMaxPointsY))
	return localMaxPoints

def descripter_funct(Points, OriginalImage, buff, plot):

	# Finds simple descriptors based on the colours
	# INPUTS: Coordinates of Interest Points, Original Image to store the RGB colors, size of the window
	# OUTPUTS: ColorHist[0:nbInterestPoints][0:3] (ColorHist[i][j]: histogram of color j for interest point i, for j 0:blue, 1:green, 2:red)

	lengthA = (buff-1)//2
	lengthB = (buff+1)//2
	boxImg = np.ones((buff,buff))
	imgread = cv2.imread('Photos/' + OriginalImage)
	img = cv2.split(imgread)
	# colorHist = []
	blueHist = []
	greenHist = []
	redHist = []

	for i in range(len(Points[0])):
		# colorHist.append([[],[],[]])
		blueHist.append([])
		greenHist.append([])
		redHist.append([])
		mask = np.zeros(imgread.shape[:2], np.uint8)
		mask[Points[0][i]-lengthA:Points[0][i]+lengthB,Points[1][i]-lengthA:Points[1][i]+lengthB] = 255
		color = ('b','g','r')
		for j,col in zip(img, color):
			# colorHist[i][idx] = []
			histr = cv2.calcHist([j],[0],mask,[256],[0,256])
			for k in range(len(histr)):
				if col == 'b':
					blueHist[i].append(histr[k][0])
				if col == 'g':
					greenHist[i].append(histr[k][0])
				if col == 'r':
					redHist[i].append(histr[k][0])
				# colorHist[i][idx].append(histr[j][0])

		if plot == 1:
			plt.figure()
			idx = 0
			for j,col in zip(img, color):
				plt.plot(colorHist[i][idx], color = col)
				idx += 1
			plt.show()

	colorHist = (np.asarray(blueHist), np.asarray(greenHist), np.asarray(redHist))
	return colorHist

def hog(img, Ix, Iy, Points, buff, plot):
	# Function to compute the histogram of gradient orientation of each interest point
	# INPUTS: Intensity derivatives, interest point coordinates
	# OUTPUT: Histogram of gradient orientation for each interest points (# Interest Points x 9 matrix)
	# PLOTS: Gradient Magnitude, Gradient Orientation and HOG for 9 random interest points

	# Compute the magnitude of the gradient
	gradMagnitude = (Ix*2+Iy2)*(1/2)
	
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
				gradOrientation[i][j] = np.arctan2(Iy[i][j],Ix[i][j])
				# Arctan returns angles between -pi/2 and pi/2, but we want only positive orientation
				# By adding pi to the negative values we have the same direction
				# if gradOrientation[i][j] < 0:
				# 	gradOrientation[i][j] = gradOrientation[i][j] + np.pi

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
	
	# 0 - Initialisation
	nbBin = 36
	sizeBin = 360/nbBin
	histOrientGrad = np.zeros((len(Points[0]),nbBin))
	lengthA = (buff-1)//2
	lengthB = (buff+1)//2

	for i in range(len(Points[0])):
	# 1 - Extract the buff x buff submatrix of magnitude and orientation
		boxMagn = gradMagnitude[Points[0][i]-lengthA:Points[0][i]+lengthB][:,Points[1][i]-lengthA:Points[1][i]+lengthB]
		boxOrient = gradOrientation[Points[0][i]-lengthA:Points[0][i]+lengthB][:,Points[1][i]-lengthA:Points[1][i]+lengthB]
		# print(boxOrient*180/np.pi)
		# plt.figure()
		# plt.imshow(img, cmap='gray')
		# plt.scatter(Points[1][i], Points[0][i], color='r', marker='+')
		# plt.scatter(Points[1][i]-lengthA, Points[0][i]-lengthA, color='r', marker='o')
		# plt.scatter(Points[1][i]+lengthB, Points[0][i]-lengthA, color='r', marker='o')
		# plt.scatter(Points[1][i]-lengthA, Points[0][i]+lengthB, color='r', marker='o')
		# plt.scatter(Points[1][i]+lengthB, Points[0][i]+lengthB, color='r', marker='o')
		# plt.show()
	# 2 - Compute the nbBin histogram for the buff x buff submatrix (0: 0, 1:1*sizeBin, ...)
		for j in range(buff):
			for k in range(buff):
				# Save the magnitude and orientation of each point in the buff x buff submatrix
				magn = boxMagn[j][k]
				orient = boxOrient[j][k]
				# Find the corresponding indices in the histogram for the point orientation
				idxMin = int(orient/(rad(sizeBin)) + nbBin/2)
				idxSup = idxMin + 1
				if idxSup > nbBin-1 and idxMin > nbBin-1:
					idxSup = idxSup-nbBin
					idxMin = idxMin-nbBin
				elif idxSup > nbBin-1:
					idxSup = idxSup - nbBin
				# Find the percentage repartition of the magnitude between the two bins
				percSup = (orient-(-np.pi+idxMin*rad(sizeBin)))/rad(sizeBin)
				percMin = 1-percSup
				if orient == np.pi:
					percSup = 0
					percMin = 1
				# Append the weighted magnitude to each bin
				histOrientGrad[i][idxMin] += percMin*magn
				histOrientGrad[i][idxSup] += percSup*magn
		del boxMagn
		del boxOrient

	# Plotting
	if plot == 1:
		plotList = random.sample(range(len(histOrientGrad)), 9)
		plt.figure()
		for i in range(9):
			idx = 330 + i + 1
			plt.subplot(idx), plt.bar(range(nbBin), histOrientGrad[plotList[i]])
			# plt.xticks(range(9), ('0', '20', '40', '60', '80', '100', '120', '140', '160'))
			plt.title(plotList[i])
		plt.suptitle('Histogram of Gradient for 9 random 8x8 cells')
		plt.show()

	return histOrientGrad

def rad(degree):

	# Function to transform a degree angle in a radians
	radian = degree*np.pi/180
	return radian

def knn(typeMat, imgBase, imgTest, matBase, matTest, pointBase, pointTest, plot):
	# Function to compute the matching interest point between two images using the HOG as a descriptor
	# INPUTS: full hog of the base image and test image (we are trying to match test with base)
	# OUTPUTS: list of nearest neighbour : i-th line is the index of the closest neighbour in Base of the i-th interest point of Test

	indexNN = []
	distanceNN = []

	for i in range(len(matTest)):
		if typeMat == "hog":
			# Store the hog of the descriptor we want to compare
			distance = np.linalg.norm(matBase-matTest[i], axis=1)
		elif typeMat == "color":
			# Compute the distance for each color and then combine it
			distance = [[],[],[]]
			for j in range(3):
				distance[j] = np.linalg.norm(matBase[j]-matTest[j][i], axis=1)
			distance = sum(distance)
		# Look for minimal distance and save the index
		minDistanceIdx = np.where(distance == np.amin(distance))
		indexNN.append(minDistanceIdx[0][0])
		distanceNN.append(np.amin(distance))

	print(indexNN)

	# Looking for the 10 best matching descriptors
	distanceMax = np.amax(distanceNN)
	minDistIdxNN = []
	for i in range(30):
		# Looking for the index of the nearest neigbour (= minimal distance)
		index = np.where(distanceNN == np.amin(distanceNN))
		index = index[0][0]
		# Set its distance to max distance so it is not taken twice for a neighbour
		distanceNN[index] = distanceMax
		# Saves its indices
		minDistIdxNN.append(index)

	pointTestX = pointTest[0]
	pointTestY = pointTest[1]
	pointBaseX = pointBase[0]
	pointBaseY = pointBase[1]
	plotBase = [[],[]]

	if plot == 1:
		# Plot the 10 best matching descriptors
		plotList = minDistIdxNN
		plotTest = [pointTestX[plotList], pointTestY[plotList]]
		for i in plotList:
			index = indexNN[i]
			plotBase[0].append(pointBaseX[index])
			plotBase[1].append(pointBaseY[index])
		# colors = ['yellow', 'red','gold', 'chartreuse', 'lightseagreen', 'darkturquoise', 'navy', 'mediumpurple', 'darkorchid', 'white']
		plt.subplot(121), plt.imshow(imgBase, cmap='gray')
		for i in range(len(minDistIdxNN)):
			plt.scatter(plotBase[0][i], plotBase[1][i], marker='+')
		plt.subplot(122), plt.imshow(imgTest, cmap='gray')
		for i in range(len(minDistIdxNN)):
			plt.scatter(plotTest[0][i], plotTest[1][i], marker='+')
		plt.show()

	return indexNN


# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

Quick1 = (['chess.png', 'chess2.png', 'chess3.png'])
Quick2 = (['chess.png', 'chess.jpg'])

allIntensity = []
allPoints = []
allDesc = []
test = Test_images
windowSize = 21 #WARNING : Must be uneve

for i in range(2):

	print("New image")
	image = test[i]
	intensity, shift = getImageIntensity(image)

	print("Computing Intensity derivatives")
	Ix, Iy = derivatives(intensity, shift, 0)
	sigma = 1.6*shift
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)

	print("Identifying corners and edges")
	CornerPoints = cornerness_funct(intensity, GIxx, GIyy, GIxy, shift, 0.05, windowSize, 0)

	print("Computing RGB descriptor")
	desc = descripter_funct(CornerPoints, image, windowSize, 0)
	
	# print("Computing histogram of gradient orientation")
	# desc = hog(intensity, Ix, Iy, CornerPoints, windowSize, 0)

	print("Saving all values")
	allDesc.append(desc)
	allIntensity.append(intensity)
	allPoints.append(CornerPoints)

AD = np.asarray(allDesc)
print(AD[1].shape)
print("Looking for matching descriptors")
u = knn("color", allIntensity[0], allIntensity[1], allDesc[0], allDesc[1], allPoints[0], allPoints[1], 1)

# # allHOG = np.array(allHOG)
# # np.savetxt('hogQuick', allHOG)
 