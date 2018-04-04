import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import time

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
	# descriptor = []
	# descriptor_mean_inten = []
	descriptor_mean_inten = np.zeros([WSplits, HSplits])

	for i in range(0,WSplits):
		# descriptor_mean_inten.append([])
		for ii in range(0,HSplits):
			# create a mask
			mask = np.zeros(img.shape[:2], np.uint8)
			mask[int((i)*width/WSplits):int((i+1)*width/WSplits), int((ii)*height/HSplits):int((ii+1)*height/HSplits)] = 255
			masked_img = cv2.bitwise_and(img,img,mask = mask)
			hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
			# descriptor.append(hist_mask)
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
			# descriptor_mean_inten[i].append(int(meanInten))
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
	intensityShiftX = np.roll(intensity, 1, axis=1)
	intensityShiftY = np.roll(intensity, 1, axis=0)

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

	cornerness = Ixx*Iyy - Ixy**2 - alpha*(Ixx + Iyy)**2

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

	# Find the intensity of the image. Returns list of lists of hold each section of the images average intensity
	# mean_intensity, WSplits, HSplits = findIntensity(FD[i], 100, 0, 0)
	#print(mean_intensity)
	# Ix, Iy = derivatives(mean_intensity, 1)

	# Create function which calculates and plots the derivatives Ix and Iy

	# Create function which calculates and plots the square of the derivatives Ix^2
	# Iy^2 and IxIy
	
	# Create a function which applies a gaussian filter to these images above
	
	# Create a cornerness function (see slide 48 of mlcv_featuredetection)

for i in range(0,1):

	mean_intensity, WSplits, HSplits = findIntensity(HD[i], 50, 0, 0)
# 	print(mean_intensity)

	Ix, Iy = derivatives(mean_intensity, 1)

	cornerness = cornerness_funct(Ix, Iy, 0.07)

	# Create function which calculates and plots the square of teh derivatives Ix^2
	# Iy^2 and IxIy
	
	# Create a function which applies a gaussian filter to these images above
	
	# Create a cornerness function (see slide 48 of mlcv_featuredetection)
