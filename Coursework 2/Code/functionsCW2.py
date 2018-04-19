# ------------------------- Packages --------------------------------
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import signal
import time
import random
import tkinter as tk
from PIL import ImageTk, Image
import os

# ------------------------- Images --------------------------------
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
		# Save the coordinates of the points for which we are currently looking for its nearest neighbour
		Point0X = PointsX[i]
		Point0Y = PointsY[i]
		# Compute the distance between each point and Point0
		distance = np.linalg.norm([PointsX-Point0X,PointsY-Point0Y], axis=0)
		# Set its own distance value to max distance so that it is not taken for a nearest neighbour
		distanceMax = np.amax(distance)
		distance[i] = distanceMax

		# Looking for the maxima R among the cornerPoint NN nearest neighbour 
		Rmax = R[Point0Y][Point0X]
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
				if abs(R[Y[k]][X[k]]) > Rmax: 
					Rmax = abs(R[Y[k]][X[k]]) 
					Xmax = X[k]
					Ymax = Y[k]

		# Save the new local maxima point if it is not already in the list
		isX = np.where(localMaxPointsX == Xmax) # indices of the local maxima with same X value
		isY = np.where(localMaxPointsY == Ymax) # indices of the local maxima with same Y value
		isAlreadyIn = []
		for j in range(len(isY[0])):
			where = np.where(isX[0] == isY[0][j])
			isAlreadyIn.append(where[0])
		# isAlreadyIn = np.where(isX[0] == isY[0]) # matching indices between isX and isY
		# if isX and isY indices matches somewhere, then it means the local maxima coordinates are already in the list
		if len(isAlreadyIn) == 0:
			localMaxPointsX.append(Xmax)
			localMaxPointsY.append(Ymax)

	localMaxPoints = (np.asarray(localMaxPointsX), np.asarray(localMaxPointsY))
	return localMaxPoints

# ------------------------- Descriptors --------------------------------
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

def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2

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
	print(pointBase)
	print(pointTest)

	indexNN = []
	distanceNN = []
		
	if typeMat == "hog":
		for i in range(len(matTest)):
			# Store the hog of the descriptor we want to compare
			distance = np.linalg.norm(matBase-matTest[i], axis=1)
			# Look for minimal distance and save the index
			index = np.where(distance == np.amin(distance))
			indexNN.append(index[0][0])
			distanceNN.append(np.amin(distance))
	
	elif typeMat == "color":
		for i in range(len(matTest[0])):
			# Compute the distance for each color and then combine it
			distance = [[],[],[]]
			for j in range(3):
				distance[j] = np.linalg.norm(matBase[j]-matTest[j][i], axis=1)
			distance = sum(distance)
			secondMin = second_smallest(distance)
			# if np.amin(distance)/secondMin <= 0.98:
				# Look for minimal distance and save the index
			index = np.where(distance == np.amin(distance))
			indexNN.append(index[0][0])
			distanceNN.append(distance[index[0][0]])
	# Looking for the best matching descriptors
	distanceMax = np.amax(distanceNN)
	interestPointsTest = [[],[]]
	interestPointsBase = [[],[]]

	for i in range(20):
		# Looking for the index of the nearest neigbour (= minimal distance)
		index = np.where(distanceNN == np.amin(distanceNN))
		index = index[0][0]
		# Set its distance to max distance so it is not taken twice for a neighbour
		distanceNN[index] = distanceMax
		# Saves its indices
		interestPointsTest[0].append(pointTest[0][index])
		interestPointsTest[1].append(pointTest[1][index])
		interestPointsBase[0].append(pointBase[0][indexNN[index]])
		interestPointsBase[1].append(pointBase[1][indexNN[index]])

	if plot == 1:
		# Plot the best matching descriptors
		colors = ['yellow', 'red','gold', 'chartreuse', 'lightseagreen', 
				  'darkturquoise', 'navy', 'mediumpurple', 'darkorchid', 'white'
				  'magenta', 'black','coral', 'orange', 'ivory',
				  'salmon','silver','teal','orchid','plum']
		plt.subplot(121), plt.imshow(imgBase, cmap='gray')
		for i in range(len(interestPointsBase[0])):
			plt.plot(interestPointsBase[0][i], interestPointsBase[1][i], marker='+')
		plt.subplot(122), plt.imshow(imgTest, cmap='gray')
		for i in range(len(interestPointsTest[0])):
			plt.plot(interestPointsTest[0][i], interestPointsTest[1][i], marker='+')

	interestPointsBase = (np.asarray(interestPointsBase[0]), np.asarray(interestPointsBase[1]))
	interestPointsTest = (np.asarray(interestPointsTest[0]), np.asarray(interestPointsTest[1]))

	return indexNN, interestPointsBase, interestPointsTest

# ------------------------- Homography and fundamental matrix --------------------------------
def findHomography(Image1, Image2, ImageA, ImageB):

	img1 = cv2.imread('Photos/' + Image1)
	img2 = cv2.imread('Photos/' + Image2)
	width, height, channels = img1.shape
	width2, height2, channels = img2.shape
	ImageA = np.asarray(ImageA).T
	ImageB = np.asarray(ImageB).T

	#set length of P matrix
	nbPoints = len(ImageA)
	P = np.zeros((2*nbPoints + 1, 9))

	#populate P matrix
	for i in range(1,nbPoints+1):
		P[(2*i-2):(2*i)][:] = np.array([[-ImageA[i-1][0], -ImageA[i-1][1], -1,             0,             0,  0, ImageA[i-1][0]*ImageB[i-1][0], ImageA[i-1][1]*ImageB[i-1][0] , ImageB[i-1][0]],
		                      			 [             0,             0,  0, -ImageA[i-1][0], -ImageA[i-1][1], -1, ImageA[i-1][0]*ImageB[i-1][1], ImageA[i-1][1]*ImageB[i-1][1] , ImageB[i-1][1]]])
	P[2*len(ImageA)][:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

	#Perform SVD
	U, S, VT = np.linalg.svd(P)
	V = VT.T

	# Set H as the last column of V (last row of VT) as it will cause least error
	H = np.zeros((3,3))
	H = V[:,-1]/V[-1,-1]
	H = H.reshape((3,3))

	# Find and print a test point to check it's working
	pointsImageA = np.concatenate((ImageA, np.ones((nbPoints,1))), axis = 1)
	point_estimated_prime = np.dot(H, pointsImageA.T).T
	points_estimated = (point_estimated_prime[:][:,0:2].T / point_estimated_prime[:][:,-1]).T

	dist_diff = np.linalg.norm(ImageB-points_estimated, axis = 1)
	Homography_accuracy = np.mean(dist_diff)

	HInv = np.linalg.inv(H)
	im_dst = cv2.warpPerspective(img2, HInv, (height, width))
	im_dst2 = cv2.warpPerspective(img1, H, (height2, width2))

	plt.figure()
	plt.subplot(2,2,1), plt.imshow(img1)
	plt.scatter(ImageA[:,0], ImageA[:,1], color='b', marker='+')
	plt.subplot(2,2,2), plt.imshow(img2)
	plt.scatter(points_estimated[:,0], points_estimated[:,1], color='r')
	plt.scatter(ImageB[:,0], ImageB[:,1], color='b', marker='+')
	plt.subplot(2,2,3), plt.imshow(im_dst)
	plt.subplot(2,2,4), plt.imshow(im_dst2)

def findFundamental(Image1, Image2, ImageA, ImageB):

	img1 = cv2.imread('Photos/' + Image1)
	img2 = cv2.imread('Photos/' + Image2)
	img1 = np.asarray(img1)
	img2 = np.asarray(img2)
	ImageA = np.asarray(ImageA).T
	ImageB = np.asarray(ImageB).T
	ImageA = np.concatenate((ImageA, np.ones((len(ImageA),1))), axis=1)
	ImageB = np.concatenate((ImageB, np.ones((len(ImageB),1))), axis=1)

	shape = img1.shape

	nbPoints = len(ImageA)
	chi = np.zeros((nbPoints, 9))

	#populate chi matrix
	for i in range(0,nbPoints):
		chi[i][:] = [ImageA[i,0]*ImageB[i,0], ImageA[i,0]*ImageB[i,1], ImageA[i,0], ImageA[i,1]*ImageB[i,0], ImageA[i,1]*ImageB[i,1], ImageA[i,1], ImageB[i,0], ImageB[i,1], 1]

	U, S, V = np.linalg.svd(chi)
	F = V.T[:,-1].reshape(3,3)/V[-1][-1]
	detF = np.linalg.det(F)

	FU, FD, FV = np.linalg.svd(F)
	FV = FV.T
	FD = np.diagflat(FD)
	FD[-1][-1] = 0
	F = np.dot(FU, np.dot(FD,FV.T))

	plt.figure()
	plt.subplot(2,1,1), plt.imshow(img1)
	plt.subplot(2,1,2), plt.imshow(img2)

	colour = ['yellow', 'red','gold', 'chartreuse', 'lightseagreen', 
			  'darkturquoise', 'navy', 'mediumpurple', 'darkorchid', 'white',
			  'magenta', 'black','coral', 'orange', 'ivory',
			  'salmon','silver','teal','orchid','plum',
			  'goldenrod','green','lightgreen','lavendar','lime']

	for i in range(0,20):

		# Finding epipolar line on image 1
		epipole1 = FV.T[:,-1]
		epipole1 = epipole1/epipole1[-1]
		epipole_x = np.arange(2*shape[0])
		epipole_y = ImageA[i,1] + (epipole_x - ImageA[i,0])*(epipole1[1]-ImageA[i,1])/(epipole1[0]-ImageA[i,0])

		# Finding epipolar line on image 2
		Epipolar = np.dot(F, ImageA[i,:].T)
		Epipolar_x = np.arange(2*shape[0])
		Epipolar_y = (-Epipolar[2] - Epipolar[0]*Epipolar_x)/Epipolar[1]

		# Plotting epipolar lines onto images
		plt.subplot(2,2,1), plt.plot(ImageA[i,0], ImageA[i,1], '+', color=colour[i])
		plt.plot(epipole_x, epipole_y, color=colour[i])
		plt.axis([0, shape[1], shape[0], 0])
		plt.subplot(2,2,2), plt.plot(ImageB[i,0], ImageB[i,1], '+', color=colour[i])
		plt.plot(Epipolar_x, Epipolar_y, color=colour[i])
		plt.axis([0, shape[1], shape[0], 0])

# ------------------------- Others --------------------------------
def rad(degree):

	# Function to transform a degree angle in a radians
	radian = degree*np.pi/180
	return radian