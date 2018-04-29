# ------------------------- Packages --------------------------------
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import NoNorm
from scipy import ndimage
from scipy import signal
import time
import random
import tkinter as tk
from PIL import ImageTk, Image
import os
from math import *
from itertools import groupby
from matplotlib.patches import ConnectionPatch
import pylab
from mpl_toolkits.mplot3d import Axes3D

# -------------------- Aggregated Functions -----------------------

def getCornerPoints(image, i, alpha, method, implemented, cornerDetectionType, descriptorType, windowSize, FAST_S, FAST_radius, FAST_threshold, maxima_NN, maxima_perc):

	intensity, imgPlot, shift = getImageIntensity(image)

	allIntensity = []
	allPoints = []
	allDesc = []
	
	if method == 'Manual':
		print("Manually find interest Points")
		Ix, Iy = derivatives(intensity, shift, 0)
		CornerPoints = manualCornerPoints(image, i)
		CornerPoints = cleanSides(intensity, CornerPoints, windowSize)
		CornerPoints = (CornerPoints[1], CornerPoints[0])

	elif method == 'Auto':
		print("Automatically find interest Points")
		print("Computing Intensity derivatives")

		if implemented == 'Implemented':
			if cornerDetectionType == 'Harris' or cornerDetectionType == 'ST':
				print("Computing Harris Corner Detector")
				Ix, Iy = derivatives(intensity, shift, 0)
				sigma = 1.6*shift
				GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)
				print("Identifying corners and edges")
				CornerPoints = cornerness_funct(intensity, imgPlot, GIxx, GIyy, GIxy, shift, alpha, windowSize, 1, maxima_NN, maxima_perc, cornerDetectionType)

			elif cornerDetectionType == 'FAST':
				print("Computing 'FAST' Corner Detector")
				Ix, Iy = derivatives(intensity, shift, 0)
				CornerPoints = FASTdetector(intensity, imgPlot, FAST_radius, FAST_S, FAST_threshold)
				CornerPoints = cleanSides(intensity, CornerPoints, windowSize)
				CornerPoints = (CornerPoints[1], CornerPoints[0])
		
		elif implemented == 'ToolBox':
			Ix, Iy = derivatives(intensity, shift, 0)
			CornerPoints = CornerTB(image, cornerDetectionType, alpha, FAST_threshold)
			CornerPoints = cleanSides(intensity, CornerPoints, windowSize)
			CornerPoints = (CornerPoints[1], CornerPoints[0])

	if descriptorType == 'RGB':
		print("Computing RGB descriptor")
		desc = rgb(CornerPoints, image, windowSize, 0)

	elif descriptorType == 'HOG':	
		print("Computing histogram of gradient orientation")
		desc = hog(intensity, Ix, Iy, CornerPoints, windowSize, 0)

	elif descriptorType == 'RGBHOG':
		print("Computing RGB histogram of gradient orientation")
		desc = np.zeros((len(CornerPoints[0]), 108))
		img = cv2.imread('Photos/' + image)
		red = img[:,:,2]
		green = img[:,:,1]
		blue = img[:,:,0]
		count = 0
		for j in [blue, green, red]:
			Ix, Iy = derivatives(j, shift, 0)
			desccol = hog(j, Ix, Iy, CornerPoints, windowSize, 0)
			desc[:,18*count:18*(count+1)] = desccol
			count += 1

	return desc, intensity, CornerPoints

# ---------------------- Toolbox Functions -----------------------

def CornerTB(image, type, alpha, FAST_threshold):

	img = cv2.imread('Photos/' + image)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

	# Inputs - image, number of corners to detect, quality of coners, min dist between corners 
	if type == 'Harris':
		print("Computing ToolBox Harris Corner Detector")
		corners = cv2.goodFeaturesToTrack(gray,1000,0.04,10, useHarrisDetector=True, k=alpha)
	elif type == 'ST':
		print("Computing ToolBox Shi-Tomasi Corner Detector")
		corners = cv2.goodFeaturesToTrack(gray,1000,0.04,10)
	elif type == 'FAST':
		print("Computing ToolBox FAST Corner Detector")
		# Initiate FAST object with default values
		corners = []
		fast = cv2.FastFeatureDetector_create(threshold=FAST_threshold)
		# find and draw the keypoints
		kp = fast.detect(img,None)
		print("Threshold: ", fast.getThreshold())
		print("nonmaxSuppression: ", fast.getNonmaxSuppression())
		print("neighborhood: ", fast.getType())
		print("Total Keypoints with nonmaxSuppression: ", len(kp))
		# Rearrange the value to fit our syntax
		for i in range(len(kp)):
			corners.append([np.asarray(kp[i].pt)])

	corners = np.int0(corners)

	cornerPointsX1 = np.asarray(corners[:,:,0])
	cornerPointsY1 = np.asarray(corners[:,:,1])
	cornerPointsX = np.asarray(cornerPointsX1[:,0])
	cornerPointsY = np.asarray(cornerPointsY1[:,0])
	CornerPoints = (cornerPointsY, cornerPointsX)

	for i in corners:
	    x,y = i.ravel()
	    cv2.circle(img,(x,y),3,255,-1)

	plt.figure()
	plt.imshow(img)
	if type == 'Harris':
		plt.title("Inbuilt Harris: Detection of Corners and Edges")
	elif type == 'ST':
		plt.title("Inbuilt Shi-Tomasi: Detection of Corners and Edges")

	return CornerPoints

# ------------------------- Images --------------------------------
def getImageIntensity(image):

	# Load an color image in grayscale
	img = cv2.imread('Photos/' + image)
	imgPlot = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = np.asarray(img)

	# Autonomously find the size of the shift window adapted to the image
	width, height = img.shape
	shift_w = int(width/100)
	shift_h = int(height/100)
	shift = max(5, min(shift_h, shift_w)) # In case the image is smaller than 50x50, shift = 3

	return img, imgPlot, shift

def getConsec(arr):
 
	count = 0
	result = 0

	for i in range(0, len(arr)):
		if (arr[i] == 0):
			count = 0
		else:
			count+= 1
			result = max(result, count) 
		 
	return result 

def get_circle(centre, radius):

	x0 = centre[0]
	y0 = centre[1]

	x, y, p = 0, radius, 1-radius

	L = []
	L.append((x, y))

	for x in range(int(radius)):
		if p < 0:
			p = p + 2 * x + 3
		else:
			y -= 1
			p = p + 2 * x + 3 - 2 * y

		L.append((x, y))

		if x >= y: break

	N = L[:]
	for i in L:
		N.append((i[1], i[0]))

	L = N[:]
	for i in N:
		L.append((-i[0], i[1]))
		L.append((i[0], -i[1]))
		L.append((-i[0], -i[1]))

	N = []
	for i in L:
		N.append((x0+i[0], y0+i[1]))

	return N

def mouse_click(event):
    # retrieve XY coords as a tuple
    x = event.x
    y = event.y
    coords = (x,y)
    # coords = root.winfo_pointerxy()
    print('coords: {}'.format(coords))
    print('X: {}'.format(coords[0]))
    print('Y: {}'.format(coords[1]))

def manualCornerPoints(image, i):

	def mouse_click(event):
	    # retrieve XY coords as a tuple
	    x = event.x
	    y = event.y
	    coords = (x,y)
	    # coords = root.winfo_pointerxy()
	    print('coords: {}'.format(coords))
	    print('X: {}'.format(coords[0]))
	    print('Y: {}'.format(coords[1]))
	    CornerPointsX.append(y)
	    CornerPointsY.append(x)
	    file.write('{}, {} \n'.format(x, y))

	CornerPointsX = []
	CornerPointsY = []

	file = open("Image" + str(i) + ".csv", 'w')
	path = 'Photos/' + image
	root = tk.Tk()
	img = Image.open(path).convert('LA')
	# img = img.resize((600, 600), Image.ANTIALIAS)
	img = ImageTk.PhotoImage(img)
	panel = tk.Label(root, image = img)
	panel.pack(side = "bottom", fill = "both", expand = "yes")
	root.bind('<Button>', mouse_click)
	click = root.bind('<Button>', mouse_click)
	root.mainloop()

	file.close()
	CornerPoints = (np.asarray(CornerPointsX), np.asarray(CornerPointsY))

	return CornerPoints

def FASTdetector(image, imgPlot, radius, S, threshold):

	width, height = image.shape

	wide = np.arange(radius,width-radius-1)
	high = np.arange(radius,height-radius-1)

	cornerPoints = []

	for i in wide[::1]:
		for j in high[::1]:
			testpoints = []
			pixelI = image[i][j]
			N = get_circle([i,j],radius)
			N = list(set(N))
			testparam = len(N)//4
			N = np.asarray(N)
			angles = []
			for n in range(0,len(N)):
				angle = atan2(N[n,0] - i, N[n,1] - j)
				angles.append(angle)
			sortedIdx = np.argsort(angles)
			N = N[sortedIdx]
			zN = list(zip(*N))

			if (pixelI + threshold > image[(zN[0][0], zN[1][0])] > pixelI - threshold and 
				pixelI + threshold > image[(zN[0][2*testparam], zN[1][2*testparam])] > pixelI - threshold):
					continue

			testpoints.append((zN[0][0], zN[1][0]))
			testpoints.append((zN[0][testparam], zN[1][testparam]))
			testpoints.append((zN[0][2*testparam], zN[1][2*testparam]))
			testpoints.append((zN[0][3*testparam], zN[1][3*testparam]))
			testpoints = list(zip(*testpoints))
			TP = np.logical_and(image[testpoints] > pixelI - threshold, image[testpoints] < pixelI - threshold)
			TP = TP.astype(int)
			if np.sum(TP) > 2:
				continue

			CircleInt = image[zN]
			greaterThan = CircleInt > pixelI + threshold
			lessThan = CircleInt < pixelI - threshold
			greaterThan = greaterThan.astype(int)
			GTmatrix = np.concatenate((greaterThan,greaterThan), axis=0)
			consecGT = getConsec(GTmatrix)
			lessThan = lessThan.astype(int)
			LTmatrix = np.concatenate((lessThan,lessThan), axis=0)
			consecLT = getConsec(LTmatrix)
			if consecGT > S or consecLT > S:
				cornerPoints.append([i, j])
	cornerPoints = np.asarray(cornerPoints)
	cornerPointsX = cornerPoints[:][:,1]
	cornerPointsY = cornerPoints[:][:,0]
	CornerPoints = (np.asarray(cornerPointsX), np.asarray(cornerPointsY))

	plt.figure()
	plt.imshow(imgPlot)
	plt.scatter(cornerPointsX, cornerPointsY, marker='+', color='red')	# plt.scatter(edgePoints[0], edgePoints[1], color='g', marker='+')
	plt.title("Detection of Corners and Edges")
	plt.show()

	return CornerPoints

# ------------------------- Corner Detection --------------------------------
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

def cornerness_funct(intensity, imgPlot, GIxx, GIyy, GIxy, shift, alpha, buff, plot, NN, perc, cornerDetectionType):

	# Function to calculate the locations of corners (and edges) in the image
	# Both Harris corner detection and Tomasi and Shi methods are implemented
	# INPUTS: Gaussian filtered intensities, alpha value for HCD, buffer to delete the sides interest points
	# OUTPUTS: Interest points coordinates
	# PLOT: Location of computed interest points on the image

	## Compute R
	if cornerDetectionType == "Harris":
		# Harris Corner detection method
		# R = (GIxx)*(GIyy) - GIxy**2 - alpha*(GIxx + GIyy)**2
		R = (GIxx*GIyy - GIxy**2)/(GIxx+GIyy+2**(-52))
	elif cornerDetectionType == "ST":
		# Tomasi and Shi method
		R = np.zeros(GIxx.shape)
		endX, endY = GIxx.shape
		for i in range(endX):
			for j in range(endY):
				M = np.array([[GIxx[i][j], GIxy[i][j]],[GIxy[i][j], GIyy[i][j]]])
				eigVals = np.linalg.eigh(M)
				R[i][j]  = np.amin(eigVals[0])

	# Based on each pixels value of R, determine if it is a corner or an edge
	# or neither. 
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
		plt.imshow(imgPlot)
		plt.scatter(cornerPoints[0], cornerPoints[1], color='r', marker='+')
		# plt.scatter(edgePoints[0], edgePoints[1], color='g', marker='+')
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
	PointsX = Points[1]
	PointsY = Points[0]
	nbPoints = len(PointsX)

	localMaxPointsX = []
	localMaxPointsY = []

	for i in range(nbPoints):
		# Compute the distance between each point and Point0
		distance = np.linalg.norm([PointsX-PointsX[i],PointsY-PointsY[i]], axis=0)
		# Set its own distance value to max distance so that it is not taken for a nearest neighbour
		distanceMax = np.amax(distance)
		distance[i] = distanceMax

		# Looking for the maxima R among the cornerPoint NN nearest neighbour 
		Rmax = R[PointsY[i]][PointsX[i]]
		Xmax = PointsX[i] 
		Ymax = PointsY[i]

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
				if abs(R[Y[k]][X[k]]) > Rmax: 
					Rmax = abs(R[Y[k]][X[k]]) 
					Xmax = X[k]
					Ymax = Y[k]

		# Save the new local maxima point if it is not already in the list and if points arent too close
		if i == 0:
			localMaxPointsX.append(Xmax)
			localMaxPointsY.append(Ymax)
		else:
			distance = np.linalg.norm((localMaxPointsX-Xmax, localMaxPointsY-Ymax), axis=0)
			distance = np.amin(distance)
			if distance > 0:
				localMaxPointsX.append(Xmax)
				localMaxPointsY.append(Ymax)

	localMaxPoints = (np.asarray(localMaxPointsX), np.asarray(localMaxPointsY))
	return localMaxPoints

# ------------------------- Descriptors --------------------------------
def rgb(Points, OriginalImage, buff, plot):

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
		blueHist.append([])
		greenHist.append([])
		redHist.append([])
		mask = np.zeros(imgread.shape[:2], np.uint8)
		mask[Points[1][i]-lengthA:Points[1][i]+lengthB,Points[0][i]-lengthA:Points[0][i]+lengthB] = 255
		color = ('b','g','r')
		for j,col in zip(img, color):
			histr = cv2.calcHist([j],[0],mask,[256],[0,256])
			for k in range(len(histr)):
				if col == 'b':
					blueHist[i].append(histr[k][0])
				if col == 'g':
					greenHist[i].append(histr[k][0])
				if col == 'r':
					redHist[i].append(histr[k][0])

		# if plot == 1:
		# 	plt.figure()
		# 	idx = 0
		# 	for j,col in zip(img, color):
		# 		plt.plot(colorHist[i][idx], color = col)
		# 		idx += 1
		# 	plt.show()

	colorHist = (np.asarray(blueHist), np.asarray(greenHist), np.asarray(redHist))
	return colorHist

def hog(img, Ix, Iy, Points, buff, plot):
	# Function to compute the histogram of gradient orientation of each interest point
	# INPUTS: Intensity derivatives, interest point coordinates
	# OUTPUT: Histogram of gradient orientation for each interest points (# Interest Points x 9 matrix)
	# PLOTS: Gradient Magnitude, Gradient Orientation and HOG for 9 random interest points

	# Compute the magnitude of the gradient
	gradMagnitude = (Ix**2+Iy**2)**(1/2)
	
	# Compute the orientation of the gradient
	endX, endY = Ix.shape
	gradOrientation = np.zeros((endX,endY))
	for i in range(endX):
		for j in range(endY):
			gradOrientation[i][j] = np.arctan2(Iy[i][j],Ix[i][j])
			if gradOrientation[i][j] < 0:
				gradOrientation[i][j] + np.pi
			
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
	nbBin = 18
	sizeBin = 180/nbBin
	histOrientGrad = np.zeros((len(Points[0]),nbBin))
	lengthA = (buff-1)//2
	lengthB = (buff+1)//2
	
	for i in range(len(Points[0])):
	# 1 - Extract the buff x buff submatrix of magnitude and orientation
		boxMagn = gradMagnitude[Points[1][i]-lengthA:Points[1][i]+lengthB][:,Points[0][i]-lengthA:Points[0][i]+lengthB]
		boxOrient = gradOrientation[Points[1][i]-lengthA:Points[1][i]+lengthB][:,Points[0][i]-lengthA:Points[0][i]+lengthB]
	# 2 - Compute the nbBin histogram for the buff x buff submatrix (0: 0, 1:1*sizeBin, ...)
		for j in range(buff):
			for k in range(buff):
				# Save the magnitude and orientation of each point in the buff x buff submatrix
				magn = boxMagn[j][k]
				orient = boxOrient[j][k]
				# Find the corresponding indices in the histogram for the point orientation
				idxMin = np.mod(int(orient/(rad(sizeBin)) + nbBin/2), nbBin)
				idxSup = np.mod(idxMin + 1, nbBin)
				# Find the percentage repartition of the magnitude between the two bins
				percSup = np.mod(orient,rad(sizeBin))/rad(sizeBin)
				percMin = 1-percSup
				# Append the weighted magnitude to each bin
				histOrientGrad[i][idxMin] += percMin*magn
				histOrientGrad[i][idxSup] += percSup*magn
		histOrientGrad[i] = histOrientGrad[i]/np.sum(boxMagn)
		del boxMagn
		del boxOrient

	# Plotting
	if plot == 1:
		plotList = random.sample(range(len(histOrientGrad)), 9)
		plt.figure()
		for i in range(9):
			idx = 330 + i + 1
			plt.subplot(idx), plt.bar(range(nbBin), histOrientGrad[plotList[i]])
			plt.title(plotList[i])
		plt.suptitle('Histogram of Gradient for 9 random 8x8 cells')
		plt.show()

	return histOrientGrad

# ------------------------- Matching --------------------------------
def knn(typeMat, img, mat, point, base, test, plot):
	# Function to compute the matching interest point between two images using the HOG as a descriptor
	# INPUTS: full hog of the base image and test image (we are trying to match test with base)
	# OUTPUTS: list of nearest neighbour : i-th line is the index of the closest neighbour in Base of the i-th interest point of Test

	imgBase = img[base]
	imgTest = img[test]
	matBase = mat[base]
	matTest = mat[test]
	pointBase = point[base]
	pointTest = point[test]

	indexNN = []
	distanceNN = []
		
	if typeMat == "HOG" or typeMat == "RGBHOG":
		for i in range(len(matTest)):
			# Store the hog of the descriptor we want to compare
			distance = np.linalg.norm(matBase-matTest[i], axis=1)
			# Look for minimal distance and save the index
			index = np.where(distance == np.amin(distance))
			indexNN.append(index[0][0])
			distanceNN.append(np.amin(distance))
	
	elif typeMat == "RGB":
		for i in range(len(matTest[0])):
			# Compute the distance for each color and then combine it
			distance = [[],[],[]]
			for j in range(3):
				distance[j] = np.linalg.norm(matBase[j]-matTest[j][i], axis=1)
			distance = sum(distance)
			# Look for minimal distance and save the index
			index = np.where(distance == np.amin(distance))
			indexNN.append(index[0][0])
			distanceNN.append(distance[index[0][0]])

	# Looking for the best matching descriptors
	distanceMax = np.amax(distanceNN)
	distanceMin = np.amin(imgBase.shape)/20
	interestPointsTest = [[],[]]
	interestPointsBase = [[],[]]

	for i in range(len(pointBase[0])): #range(30):
		# Looking for the index of the nearest neigbour (= minimal distance)
		index = np.where(distanceNN == np.amin(distanceNN))
		index = index[0][0]
		# Set its distance to max distance so it is not taken twice for a neighbour
		distanceNN[index] = distanceMax
		
		if i == 0:
			# Saves its indices
			interestPointsTest[0].append(pointTest[0][index])
			interestPointsTest[1].append(pointTest[1][index])
			interestPointsBase[0].append(pointBase[0][indexNN[index]])
			interestPointsBase[1].append(pointBase[1][indexNN[index]])
		else:
			distance = np.linalg.norm((interestPointsBase[0]-pointBase[0][indexNN[index]], interestPointsBase[1]-pointBase[1][indexNN[index]]), axis=0)
			distance = np.amin(distance)
			if distance > distanceMin:
				interestPointsTest[0].append(pointTest[0][index])
				interestPointsTest[1].append(pointTest[1][index])
				interestPointsBase[0].append(pointBase[0][indexNN[index]])
				interestPointsBase[1].append(pointBase[1][indexNN[index]])

	if plot == 1:
		# Plot the best matching descriptors
		fig = plt.figure(7)
		colors = ['yellow', 'red','gold', 'chartreuse', 'lightseagreen', 
				  'darkturquoise', 'navy', 'mediumpurple', 'darkorchid', 'white',
				  'magenta', 'black','coral', 'orange', 'ivory',
				  'salmon','silver','teal','orchid','plum']
		idxplot = np.random.choice(len(interestPointsBase[0]), 10, 0)
		ax1 = fig.add_subplot(121)
		plt.imshow(imgBase, cmap='gray')
		ax2 = fig.add_subplot(122)
		plt.imshow(imgTest, cmap='gray')
		for i, j in zip(idxplot, range(15)):
			ax1.plot(interestPointsBase[0][i], interestPointsBase[1][i], marker='+', markersize='40', color=colors[j])
			ax2.plot(interestPointsTest[0][i], interestPointsTest[1][i], marker='+', markersize='40', color=colors[j])
			xy2 = (interestPointsBase[0][i], interestPointsBase[1][i])
			xy1 = (interestPointsTest[0][i], interestPointsTest[1][i])
			con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
		                     	  axesA=ax2, axesB=ax1, color=colors[j])
			ax2.add_artist(con)

	interestPointsBase = (np.asarray(interestPointsBase[0]), np.asarray(interestPointsBase[1]))
	interestPointsTest = (np.asarray(interestPointsTest[0]), np.asarray(interestPointsTest[1]))

	# img3 = cv2.drawMatches(imgBase,interestPointsBase,imgTest,interestPointsTest, index,None,flags=0)
	# plt.imshow(img3)

	return indexNN, interestPointsBase, interestPointsTest

# ---------------- Find homography and fundamental matrixes ------------------------

def findHomography(Image1, Image2, ImageA, ImageB, selection):

	img1 = cv2.imread('Photos/' + Image1)
	img1Plot = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
	img2 = cv2.imread('Photos/' + Image2)
	img2Plot = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
	width, height, channels = img1.shape
	width2, height2, channels = img2.shape

	ImageA = np.asarray(ImageA).T
	ImageB = np.asarray(ImageB).T
	K = int(0)
	goodPercent = int(0)
	goodPercentBest = int(0)

	while K < 500:

		fewPointsIdx = np.random.choice(len(ImageA), selection, 0)
		ImageAfew = ImageA[fewPointsIdx,:]
		ImageBfew = ImageB[fewPointsIdx,:]

		nbPoints = len(ImageAfew)
		nbPoints_all = len(ImageA)

		P = np.zeros((2*nbPoints + 1, 9))
		for i in range(1,nbPoints+1):
			P[(2*i-2):(2*i)][:] = np.array([[-ImageAfew[i-1][0], -ImageAfew[i-1][1], -1,             0,             0,  0, ImageAfew[i-1][0]*ImageBfew[i-1][0], ImageAfew[i-1][1]*ImageBfew[i-1][0] , ImageBfew[i-1][0]],
			                      			 [             0,             0,  0, -ImageAfew[i-1][0], -ImageAfew[i-1][1], -1, ImageAfew[i-1][0]*ImageBfew[i-1][1], ImageAfew[i-1][1]*ImageBfew[i-1][1] , ImageBfew[i-1][1]]])
		P[2*len(ImageAfew)][:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

		#Perform SVD
		U, S, VT = np.linalg.svd(P)
		V = VT.T

		# Set H as the last column of V (last row of VT) as it will cause least error
		H = np.zeros((3,3))
		H = V[:,-1]/V[-1,-1]
		H = H.reshape((3,3))

		# Find and print a test point to check it's working
		pointsImageA = np.concatenate((ImageAfew, np.ones((nbPoints,1))), axis = 1)
		point_estimated_prime = np.dot(H, pointsImageA.T).T
		points_estimated = (point_estimated_prime[:][:,0:2].T / point_estimated_prime[:][:,-1]).T
		dist_diff = np.linalg.norm(ImageBfew-points_estimated, axis = 1)

		pointsImageA_all = np.concatenate((ImageA, np.ones((nbPoints_all,1))), axis = 1)
		point_estimated_prime_all = np.dot(H, pointsImageA_all.T).T
		points_estimated_all = (point_estimated_prime_all[:][:,0:2].T / point_estimated_prime_all[:][:,-1]).T
		dist_diff_all = np.linalg.norm(ImageB-points_estimated_all, axis=1)
		
		acceptableIdx = np.where(dist_diff_all < 5)
		goodPercent = len(acceptableIdx[0])/len(ImageA)
		if goodPercent > goodPercentBest:
			HBest = H
			goodPercentBest = goodPercent
			fewPointsIdxBest = fewPointsIdx
			acceptableIdxBest = acceptableIdx
		if goodPercent > 0.8:
			break
		K += 1
		
	pointsImageA_all = np.concatenate((ImageA, np.ones((nbPoints_all,1))), axis = 1)
	point_estimated_prime_all = np.dot(HBest, pointsImageA_all.T).T
	points_estimated_all = (point_estimated_prime_all[:][:,0:2].T / point_estimated_prime_all[:][:,-1]).T
	dist_diff_all = np.linalg.norm(ImageB-points_estimated_all, axis=1)

	Homography_accuracy = np.mean(dist_diff_all)
	HInv = np.linalg.inv(HBest)
	im_desc = cv2.warpPerspective(img2Plot, HInv, (height, width))
	im_desc2 = cv2.warpPerspective(img1Plot, HBest, (height2, width2))

	# im_rec = cv2.cvtColor(im_desc, cv2.COLOR_RGB2GRAY)
	# im_rec2 = cv2.cvtColor(im_desc2, cv2.COLOR_RGB2GRAY)
	im_rec = im_desc
	im_rec2 = im_desc2

	scaleFactor = min(width2, height2)
	Homography_accuracy_norm = Homography_accuracy/scaleFactor*100

	print(HBest)

	plt.figure(2)
	plt.suptitle('Noramlised Homography Accuracy (Automatic Detection) = %1.2f %%' % Homography_accuracy_norm, fontsize=12)
	
	ax1 = plt.subplot(2,2,1)
	ax1.set_title('First Image: Interest Points', fontsize=12)
	ax1.imshow(img1Plot)
	ax1.scatter(ImageA[:,0], ImageA[:,1], color='b', marker='+', s=40)
	ax1.scatter(ImageA[fewPointsIdxBest,0], ImageA[fewPointsIdxBest,1], color='y', marker='*', s=40)
	plt.xlabel('Pixels', fontsize=12)
	plt.ylabel('Pixels', fontsize=12)

	ax2 = plt.subplot(2,2,2)
	ax2.set_title('Second Image: Interest Points', fontsize=12)
	ax2.imshow(img2Plot, cmap='gray')
	ax2.scatter(points_estimated_all[:,0], points_estimated_all[:,1], color='b', marker='+')
	ax2.scatter(ImageB[:,0], ImageB[:,1], color='r', marker='+', s=40)
	ax2.scatter(ImageB[fewPointsIdxBest,0], ImageB[fewPointsIdxBest,1], color='y', marker='*', s=40)
	plt.xlabel('Pixels', fontsize=12)
	plt.ylabel('Pixels', fontsize=12)

	ax3 = plt.subplot(2,2,3)
	ax3.set_title('Second Image Homography Adjusted', fontsize=12)
	ax3.imshow(im_desc, cmap='gray')
	plt.xlabel('Pixels', fontsize=12)
	plt.ylabel('Pixels', fontsize=12)

	ax4 = plt.subplot(2,2,4)
	ax4.set_title('First Image Homography Adjusted', fontsize=12)
	ax4.imshow(im_desc2, cmap='gray')
	plt.xlabel('Pixels', fontsize=12)
	plt.ylabel('Pixels', fontsize=12)

	plt.figure()
	plt.suptitle('Inliers/Outliers', fontsize=16)
	idxplot = np.random.choice(len(ImageA), 12, 0)
	ax1 = plt.subplot(121)
	ax1.imshow(img1Plot, cmap='gray')
	plt.xlabel('Pixels', fontsize=14)
	plt.ylabel('Pixels', fontsize=14)
	ax2 = plt.subplot(122)
	ax2.imshow(img2Plot, cmap='gray')
	plt.xlabel('Pixels', fontsize=14)
	for i in idxplot:
		if i in acceptableIdxBest[0]:
			ax1.plot(ImageA[i][0], ImageA[i][1], marker='+', markersize='10', color='lightgreen')
			ax2.plot(ImageB[i][0], ImageB[i][1], marker='+', markersize='10', color='lightgreen')
			xy2 = (ImageA[i][0], ImageA[i][1])
			xy1 = (ImageB[i][0], ImageB[i][1])
			con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
		                     	  axesA=ax2, axesB=ax1, color='lightgreen')
			ax2.add_artist(con)
		else:
			ax1.plot(ImageA[i][0], ImageA[i][1], marker='+', markersize='10', color='k')
			ax2.plot(ImageB[i][0], ImageB[i][1], marker='+', markersize='10', color='k')
			xy2 = (ImageA[i][0], ImageA[i][1])
			xy1 = (ImageB[i][0], ImageB[i][1])
			con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
		                     	  axesA=ax2, axesB=ax1, color='k')
			ax2.add_artist(con)

	return ImageAfew, ImageBfew, HBest, Homography_accuracy, Homography_accuracy_norm, im_rec, points_estimated_all.T

def findFundamental(Image1, Image2, ImageA, ImageB):

	img1 = cv2.imread('Photos/' + Image1)
	img1Plot = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
	img1 = np.asarray(img1)
	try:
		img2 = cv2.imread('Photos/' + Image2)
		img2Plot = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
		img2 = np.asarray(img2)
	except:
		img2 = Image2
		img2Plot = img2

	ImageA = np.asarray(ImageA).T
	ImageB = np.asarray(ImageB).T
	ImageA = np.concatenate((ImageA, np.ones((len(ImageA),1))), axis=1)
	ImageB = np.concatenate((ImageB, np.ones((len(ImageB),1))), axis=1)

	shape = img1.shape
	shape2 = img2.shape

	K = int(0)
	fundamentalAccuracyBest = int(100000)

	while K < 500:

		resTot = int(0)
		fewPointsIdx = np.random.choice(len(ImageA), 20, 0)
		ImageAfew = ImageA[fewPointsIdx,:]
		ImageBfew = ImageB[fewPointsIdx,:]

		nbPoints = len(ImageAfew)
		chi = np.zeros((nbPoints, 9))

		#populate chi matrix
		for i in range(0,nbPoints):
			chi[i][:] = [ImageAfew[i,0]*ImageBfew[i,0], ImageAfew[i,1]*ImageBfew[i,0], ImageBfew[i,0], ImageAfew[i,0]*ImageBfew[i,1], ImageAfew[i,1]*ImageBfew[i,1], ImageBfew[i,1], ImageAfew[i,0], ImageAfew[i,1], 1]

		U, S, VT = np.linalg.svd(chi)
		V = VT.T
		F = V[:,-1].reshape(3,3)/V[-1][-1]
		F = F.T
		detF = np.linalg.det(F)

		FU, FD, FVT = np.linalg.svd(F)
		FV = FVT.T
		FD = np.diagflat(FD)
		FD[-1][-1] = 0
		F = np.dot(FU, np.dot(FD,FV.T))
		resMin = []
		resMinIdx = []
		distMin = []
		distMinIdx = []
		distanceTotal = int(0)

		for i in range(len(ImageA)):

		# 	res = np.dot(ImageA[i,:].T,np.dot(F,ImageB[i,:]))
		# 	resTot += abs(res)

		# 	if i < 6:
		# 		resMinIdx.append(i)
		# 		resMin.append(res)
		# 	elif res < np.amax(resMin):
		# 		maxidx = np.where(resMin == np.amax(resMin))
		# 		resMinIdx[maxidx[0][0]] = i
		# 		resMin[maxidx[0][0]] = res
		# 		
		# fundamentalAccuracy = resTot/len(ImageA)
		
			# Finding epipolar line on image 2
			Epipolar = np.dot(F, np.transpose(ImageA[i,:]))
			dist = ((Epipolar[0]*ImageB[i,0] + Epipolar[1]*ImageB[i,1] + Epipolar[2])**2)**(1/2)/np.sqrt(Epipolar[0]**2 + Epipolar[1]**2)
			distanceTotal += dist

			if i < 6:
				distMinIdx.append(i)
				distMin.append(dist)
			elif dist < np.amax(distMin):
				maxidx = np.where(distMin == np.amax(distMin))
				distMinIdx[maxidx[0][0]] = i
				distMin[maxidx[0][0]] = dist

		fundamentalAccuracy = distanceTotal/len(ImageA)

		if fundamentalAccuracy < fundamentalAccuracyBest:
			fundamentalAccuracyBest = fundamentalAccuracy
			distBestIdx = distMinIdx
			distBest = distMin
			FBest = F
			FVBest = FV
			FUBest = FU

		K += 1

	epipole1 = FVBest.T[:,-1]
	epipole1 = epipole1/epipole1[-1]
	epipole1 = np.round(epipole1,2)
	TE1 = str(epipole1)
	print(TE1)

	epipole2 = FUBest[:,-1]
	epipole2 = epipole2/epipole2[-1]
	epipole2 = np.round(epipole2,2)
	TE2 = str(epipole2)
	print(TE2)

	plt.figure()
	plt.suptitle("Epipolar Geometry", fontsize=14)
	ax1 = plt.subplot(1,2,1)
	ax1.set_title('First Image', fontsize=14)
	plt.xlabel('Pixels', fontsize=14)
	plt.ylabel('Pixels', fontsize=14)
	ax1.imshow(img1Plot)
	ax2 = plt.subplot(1,2,2)
	ax2.set_title('Second Image', fontsize=14)
	plt.xlabel('Pixels', fontsize=14)
	plt.ylabel('Pixels', fontsize=14)
	ax2.imshow(img2Plot)

	colors = ['yellow', 'red','magenta', 'black', 'blue', 
		  'darkturquoise', 'navy', 'mediumpurple', 'darkorchid', 'white',
		  'salmon', 'chartreuse','coral', 'orange', 'ivory',
		  'gold','silver','teal','orchid','plum']

	for i, j in zip(distBestIdx, range(6)):

		# Finding epipolar line on image 1
		epipole_x = np.arange(2*shape[0])
		epipole_y = ImageA[i,1] + (epipole_x - ImageA[i,0])*(epipole1[1]-ImageA[i,1])/(epipole1[0]-ImageA[i,0])

		# Finding epipolar line on image 2
		Epipolar = np.dot(FBest, np.transpose(ImageA[i,:]))
		Epipolar_x = np.arange(2*shape[0])
		Epipolar_y = (-Epipolar[2] - Epipolar[0]*Epipolar_x)/Epipolar[1]

		# Plotting epipolar lines onto images
		plt.subplot(1,2,1), plt.plot(ImageA[i,0], ImageA[i,1], 'o', markersize=6, color=colors[j])
		plt.plot(epipole_x, epipole_y, color=colors[j])
		plt.axis([0, shape[1], shape[0], 0])
		plt.subplot(1,2,2), plt.plot(ImageB[i,0], ImageB[i,1], 'o', markersize=6, color=colors[j])
		plt.plot(Epipolar_x, Epipolar_y, color=colors[j])
		plt.axis([0, shape[1], shape[0], 0])

	scaleFactor = min(shape[0]/shape2[0], shape[1]/shape2[1])
	fundAccNorm = fundamentalAccuracyBest/scaleFactor

	return fundamentalAccuracyBest, fundAccNorm

def dispMap(Image1, Image2, windowSize, derivative, T):

	shift = 5
	# Load images in grayscale
	img1 = cv2.imread('Photos/' + Image1,0)
	img1 = np.asarray(img1)

	try:
		img2 = cv2.imread('Photos/' + Image2, 0)
		img2 = np.asarray(img2)
	except:
		img2 = Image2
		img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

	if derivative == 'Yes':
		Ix1, Iy1 = derivatives(img1, shift, 0)
		img1 = Iy1*Ix1
		Ix2, Iy2 = derivatives(img2, shift, 0)
		img2 = Iy2*Ix2

	# Initialisation
	halfWS = int((windowSize-1)/2)
	disparityMap = np.zeros(img1.shape)
	depthMap = np.zeros(img1.shape)
	height, width = img1.shape
	disparityRange = int(T[0]) #int(min(width, height)/10)

	# Looping
	for i in range(height):
		minH = max(0, i-halfWS)
		maxH = min(height, i+halfWS)
		for j in range(width):
			minW = max(0, j-halfWS)
			maxW = min(width, j+halfWS)
			minD = max(-disparityRange, -minW)
			# minD = 0
			maxD = min(disparityRange, width - maxW)
			# Select the reference block from img1
			# template = img1[minW:maxW, minH:maxH]
			template = img2[minH:maxH, minW:maxW]
			# Get the number of blocks in this search.
			numBlocks = maxD - minD
			# Create a vector to hold the block differences.
			blockDiffs = np.zeros((numBlocks, 1))
			for k in range(minD,maxD):
				block = img1[minH:maxH, minW+k:maxW+k]	
				blockIndex = k - minD
				blockDiffs[blockIndex] = np.sum(abs(template - block))
			bestMatchDisp = np.amin(blockDiffs)
			bestMatchIdx = np.where(blockDiffs == np.amin(blockDiffs))
			bestMatchIdx = bestMatchIdx[0][0]

			if bestMatchIdx == 0 or bestMatchIdx == numBlocks - 1:
				disparityMap[i,j] = bestMatchIdx + minD
			else:
				C1 = blockDiffs[bestMatchIdx-1]
				C2 = bestMatchDisp
				C3 = blockDiffs[bestMatchIdx+1]
				disparityMap[i,j] = bestMatchIdx + minD - 0.5*(C3-C1)/(C1-2*C2+C3)
			del blockDiffs

	disparityMapMin = np.amin(disparityMap)
	disparityMapMax = np.amax(disparityMap)

	for i in range(height):
		for j in range(width):
			disparityMap[i,j] = - disparityMapMin + disparityMap[i,j]*6/(disparityMapMax - disparityMapMin)
			depthMap[i,j] = 30/(disparityMap[i,j])

	print(np.amin(disparityMap), np.amax(disparityMap))	
	print(np.amin(depthMap), np.amax(depthMap))

	return disparityMap, depthMap

def stereoRectification(Image1, Image2, ImageA, ImageB, T0, R, f):

	img1 = cv2.imread('Photos/' + Image1,0)
	img1 = np.asarray(img1)
	width, height = img1.shape
	img2 = cv2.imread('Photos/' + Image2,0)
	img2 = np.asarray(img2)
	width2, height2 = img2.shape

	# Step 1
	e1 = T0/np.linalg.norm(T0)
	print(e1)
	t2 = [-T0[1], T0[0], T0[2]]
	t2 = np.asarray([t2])
	t2 = t2.T
	e2 = 1/np.linalg.norm(T0)*t2
	e2 = e2[0]
	e3 = np.cross(e1.T, e2.T).T
	Rrect = np.concatenate((np.concatenate((e1.T, e2.T), axis=0), e3.T), axis=0)
	
	# Step 2
	Rleft = Rrect
	Rright = np.dot(R,Rrect).T

	# Step 3
	ImageA = np.asarray(ImageA).T
	ImageA = np.concatenate((ImageA, f*np.ones((len(ImageA),1))), axis = 1).T
	ImageB = np.asarray(ImageB).T
	ImageB = np.concatenate((ImageB, f*np.ones((len(ImageB),1))), axis = 1).T
	points_estimatedA = np.dot(Rleft,ImageA)
	points_estimatedA = f/points_estimatedA[2]*points_estimatedA
	points_estimatedB = np.dot(Rright,ImageB)
	points_estimatedB = f/points_estimatedB[2]*points_estimatedB

	im_desc = cv2.warpPerspective(img2, Rright, (height, width))
	im_desc2 = cv2.warpPerspective(img1, Rleft, (height2, width2))

	plt.figure(11)
	plt.subplot(211), plt.imshow(im_desc)
	plt.subplot(212), plt.imshow(im_desc2)

	plt.figure(12)
	plt.subplot(2,2,1), plt.imshow(img1, cmap='gray')
	plt.scatter(ImageA[0,:], ImageA[1,:], color='b', marker='+', s=40)
	# plt.scatter(points_estimatedB[0,:],points_estimatedB[1,:], color='r', marker='o', s=40 )
	plt.subplot(2,2,3), plt.imshow(im_desc, cmap='gray')
	plt.scatter(points_estimatedB[0,:],points_estimatedB[1,:], color='r', marker='o', s=40 )
	plt.subplot(2,2,2), plt.imshow(img2, cmap='gray')
	plt.scatter(ImageB[0,:], ImageB[1,:], color='b', marker='+', s=40)
	# plt.scatter(points_estimatedA[0,:],points_estimatedA[1,:], color='r', marker='o', s=40 )
	plt.subplot(2,2,4), plt.imshow(im_desc2, cmap='gray')
	plt.scatter(points_estimatedA[0,:],points_estimatedA[1,:], color='r', marker='o', s=40 )


# ------------------------- Others --------------------------------
def rad(degree):

	# Function to transform a degree angle in a radians
	radian = degree*np.pi/180
	return radian

def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2

def rigid_transform_3D(A, B):

	A = np.asarray(A).T
	B = np.asarray(B).T
	A = np.concatenate((A, np.ones((len(A),1))), axis=1)
	B = np.concatenate((B, np.ones((len(B),1))), axis=1)

	A = A[1:5,:]
	B = B[1:5,:]

	assert len(A) == len(B)

	N = A.shape[0]; # total points

	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)

	# centre the points
	AA = A - np.tile(centroid_A, (N, 1))
	BB = B - np.tile(centroid_B, (N, 1))

	# dot is matrix multiplication for array
	H = np.dot(np.transpose(AA), BB)

	U, S, Vt = np.linalg.svd(H)

	R = np.dot(Vt.T,U.T)

	# special reflection case
	if np.linalg.det(R) < 0:
		Vt[2,:] *= -1
		R = np.dot(Vt.T,U.T)

	t = -np.dot(R,centroid_A.T) + centroid_B.T

	return R, t