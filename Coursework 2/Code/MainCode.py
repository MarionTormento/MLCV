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
	plt.title("Ix")
	plt.subplot(232), plt.imshow(Iy, cmap='gray', interpolation='nearest')
	plt.title("Iy")
	plt.subplot(233), plt.imshow(Ixx, cmap='gray', interpolation='nearest')
	plt.title("Ixx")
	plt.subplot(234), plt.imshow(Iyy, cmap='gray', interpolation='nearest')
	plt.title("Iyy")
	plt.subplot(235), plt.imshow(Ixy, cmap='gray', interpolation='nearest')
	plt.title("Ixy")

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
	# Find local maxima for the corners
	maxCornerPointsX, maxCornerPointsY = local_maxima(R, cornerPoints, 8)
	CornerPoints = (np.asarray(maxCornerPointsX), np.asarray(maxCornerPointsY))

	thresholdEdge = np.percentile(R, 3)
	edgePoints = np.where(R < thresholdEdge)
	# Find local minima for the edges
	maxEdgePointsX, maxEdgePointsY = local_maxima(R, edgePoints, 8)
	EdgePoints = (np.asarray(maxEdgePointsX), np.asarray(maxEdgePointsY))
	
	# Plot
	plt.figure()
	plt.imshow(intensity, cmap='gray')
	plt.scatter(maxCornerPointsX, maxCornerPointsY, color='r', marker='+')
	plt.scatter(maxEdgePointsX, maxEdgePointsY, color='g', marker='+')
	plt.title("Detection of Corners and Edges")
	plt.show()

	return R, CornerPoints, EdgePoints

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


# def hog(Ix, Iy, shift):
# 	# Compute the magnitude of the gradient
# 	gradMagnitude = (Ix**2+Iy**2)**(1/2)
	
# 	# Compute the orientation of the gradient
# 	endY, endX = Ix.shape
# 	gradOrientation = np.zeros((endY,endX))
# 	for i in range(endX):
# 		for j in range(endY):
# 			if Ix[j][i] == 0 and Iy[j][i] != 0:
# 				gradOrientation[j][i] = np.pi/2
# 			elif Ix[j][i] == 0 and Iy[j][i] == 0:
# 				gradOrientation[j][i] = 0
# 			else:
# 				gradOrientation[j][i] = np.arctan(Iy[j][i]/Ix[j][i])

# 	# Plotting
# 	plt.figure()
# 	plt.subplot(121), plt.imshow(gradMagnitude, cmap='gray', interpolation='nearest')
# 	plt.title("Gradient Magnitude")
# 	plt.subplot(122), plt.imshow(gradOrientation, cmap='gray', interpolation='nearest')
# 	plt.title("Gradient Magnitude")
# 	plt.show()

# 	# Calculate Histogram of Gradients in 8×8 cells
# 	# 1 - Extract the 8x8 submatrix of magnitude and orientation
# 	# 2 - Compute the 0 bin histogram for the cell (0: 0, 1:20, ..., 9:160)
# 	# https://www.learnopencv.com/histogram-of-oriented-gradients/ 

# 	cellSize = 8
# 	cellMagn = np.zeros((cellSize,cellSize))
# 	cellOrient = np.zeros((cellSize,cellSize))
# 	cellX = int(endX/cellSize)
# 	cellY = int(endY/cellSize)
# 	histOrientGrad = np.zeros((cellX*cellY,9))
# 	for i in range(cellX): 
# 		for j in range(cellY):
# 			cellMagn = gradMagnitude[cellSize*j:cellSize*(j+1)][cellSize*i:cellSize*(i+1)]			
# 			cellOrient = gradOrientation[cellSize*j:cellSize*(j+1)][cellSize*i:cellSize*(i+1)]
# 			for ii in range(cellSize):
# 				for jj in range(cellSize):
					


# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

for i in range(0,1):

	intensity, shift = getImageIntensity('chess.jpg')

	Ix, Iy = derivatives(intensity, shift)
	
	sigma = 1.6*shift
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)
	
	R, CornerPoints, EdgePoints = cornerness_funct(intensity, GIxx, GIyy, GIxy, 0.05)

	descripter_funct(CornerPoints, 'chess.jpg')
	# hog(Ix,Iy, shift)

