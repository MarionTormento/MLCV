import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import time

def getImageIntensity(image):

	print('Starting intensity find')
	t = time.time()

	# Load an color image in grayscale
	img = cv2.imread('Photos/' + image, 0)
	width, height = img.shape

	return img

def findIntensity(image, inputSmoothing, plot1, plot2):

	print('Starting intensity find')
	t = time.time()

	# Load an color image in grayscale
	img = cv2.imread('Photos/' + image, 0)
	width, height = img.shape

	# Plot image and see its full histogram
	if plot1>0:
		fig1 = plt.figure()
		plt.subplot(221), plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
		# plt.subplot(212), plt.hist(img.ravel(),256,[0,256]);

	# Define length scale over which with intensity is averaged
	smoothing = int(inputSmoothing)
	HSplits = int(height/inputSmoothing)
	WSplits = int(width/inputSmoothing)

	# Initialise stuff
	descriptor_mean_inten = np.zeros([WSplits, HSplits])

	for i in range(0,WSplits):
		for ii in range(0,HSplits):
			# create a mask
			mask = np.zeros(img.shape[:2], np.uint8)
			mask[int((i)*width/WSplits):int((i+1)*width/WSplits), int((ii)*height/HSplits):int((ii+1)*height/HSplits)] = 255
			masked_img = cv2.bitwise_and(img,img,mask = mask)
			hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
			# Plot each section of the image and its histogram
			if plot2>0:
				figure = plt.figure()
				plt.subplot(211), plt.imshow(masked_img, 'gray')
				plt.subplot(212), plt.plot(hist_mask)
				plt.xlim([0,256])
			sumInten = 0
			sumHist = 1
			# Calculate the mean intensity of the pixels in this section of the image
			for j in range(0,255):
				sumInten += j*hist_mask[j]
				sumHist += hist_mask[j]
			meanInten = int(sumInten)/int(sumHist)
			descriptor_mean_inten[i][ii] = int(meanInten)

			del mask, masked_img, hist_mask, sumHist, meanInten, sumInten

	# descriptor_mean_intenArray = np.asarray(descriptor_mean_inten) 
	#print(np.asarray(descriptor_mean_inten) is descriptor_mean_inten)
	print(descriptor_mean_inten)
	#descriptor_mean_intenArray.reshape((HSplits, WSplits))
	print(HSplits,WSplits)
	print(descriptor_mean_inten.shape)

	elapsed = time.time() - t
	print('Time to find mean intensity = % 1.0fs' % elapsed)

	return descriptor_mean_inten, WSplits, HSplits

def derivatives(intensity, shift):

	print('Starting Derivatives')
	t = time.time()

	# Find Ix and Iy
	intensity = np.asarray(intensity)
	intensityShiftX = np.roll(intensity, shift, axis=1)
	intensityShiftY = np.roll(intensity, shift, axis=0)

	Ix = np.subtract(intensityShiftX, intensity) / shift
	Iy = np.subtract(intensityShiftY, intensity) / shift
	Ixx = Ix ** 2
	Iyy = Iy ** 2
	Ixy = Ix * Iy

	elapsed = time.time() - t
	print('Time to find derivatives = % 1.0fs' % elapsed)

	print('Plotting Derivatives')
	t1 = time.time()

	figure = plt.figure()
	# plt.subplot(222), plt.imshow(intensity, cmap='gray', interpolation='nearest')
	plt.subplot(231), plt.imshow(Ix, cmap='gray', interpolation='nearest')
	plt.subplot(232), plt.imshow(Iy, cmap='gray', interpolation='nearest')
	plt.subplot(233), plt.imshow(Ixx, cmap='gray', interpolation='nearest')
	plt.subplot(234), plt.imshow(Iyy, cmap='gray', interpolation='nearest')
	plt.subplot(235), plt.imshow(Ixy, cmap='gray', interpolation='nearest')

	elapsed1 = time.time() - t1
	print('Time to plot derivatives = % 1.0fs' % elapsed1)

	del intensity

	return Ix, Iy

def cornerness_funct(Ix, Iy, alpha):

	print('Starting Cornerness')
	t = time.time()

	Ixx = Ix ** 2
	Iyy = Iy ** 2
	Ixy = Ix * Iy

	cornerness = (Ixx**2)*(Iyy**2) - Ixy**2 - alpha*(Ixx + Iyy)**2

	elapsed = time.time() - t
	print('Time to find cornerness = % 1.0fs' % elapsed)

	print('Plotting Cornerness')
	t1 = time.time()

	plt.figure()
	plt.imshow(cornerness, cmap='gray', interpolation='nearest')
	plt.show()

	elapsed1 = time.time() - t1
	print('Time to plot cornerness = % 1.0fs' % elapsed1)

	return cornerness

# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

# for i in range(0,1):

# 	# ---------------------------------------------------------

# 	# TRY THIS ONE IT'S SUPER SLOW BUT MAYBE BETTER RESULTS?
# 	# intensity, WSplits, HSplits = findIntensity(FD[i], 50, 0, 0)

# 	# TRY THIS ONE IT'S MUCH FASTER BUT WORSE?
# 	intensity = getImageIntensity(FD[i])
	
# 	# ---------------------------------------------------------

# 	Ix, Iy = derivatives(intensity, 150)
	
# 	# Create a function which applies a gaussian filter to these images above
	
# 	cornerness = cornerness_funct(Ix, Iy, 0.05)
	

for i in range(0,1):

	# ---------------------------------------------------------

	# TRY THIS ONE IT'S SUPER SLOW BUT MAYBE BETTER RESULTS?
	# intensity, WSplits, HSplits = findIntensity(HD[i], 50, 0, 0)

	# TRY THIS ONE IT'S MUCH FASTER BUT WORSE?
	intensity = getImageIntensity(HD[i])
	
	# ---------------------------------------------------------

	Ix, Iy = derivatives(intensity, 15)
	
	# Create a function which applies a gaussian filter to these images above
	
	cornerness = cornerness_funct(Ix, Iy, 0.05)
