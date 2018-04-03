import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def findIntensity(image, inputSmoothing, plot1, plot2):

	# Load an color image in grayscale
	img = cv2.imread('Photos/' + image, 0)
	width, height = img.shape

	# Plot image and see its full histogram
	if plot1>0:
		fig1 = plt.figure()
		plt.subplot(211), plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
		plt.subplot(212), plt.hist(img.ravel(),256,[0,256]);

	# Define length scale over which with intensity is averaged
	smoothing = int(inputSmoothing)
	HSplits = int(height/inputSmoothing)
	WSplits = int(width/inputSmoothing)

	# Initialise stuff
	descriptor = []
	descriptor_mean = []
	descriptor_mean_inten = []

	for i in range(0,WSplits):
		# Create a new list
		descriptor_mean_inten.append([])
		for ii in range(0,HSplits):
			# create a mask
			mask = np.zeros(img.shape[:2], np.uint8)
			mask[int((i)*width/WSplits):int((i+1)*width/WSplits), int((ii)*height/HSplits):int((ii+1)*height/HSplits)] = 255
			masked_img = cv2.bitwise_and(img,img,mask = mask)
			hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
			descriptor.append(hist_mask)
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
			descriptor_mean_inten[i].append(meanInten)

			del mask, masked_img, hist_mask, sumHist
	plt.show()

	return descriptor_mean_inten, descriptor, WSplits, HSplits
	

# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

for i in range(0,1):

	# Find the intensity of the image. Returns list of lists of hold each section of the images average intensity
	mean_intensity, image_intensity, WSplits, HSplits = findIntensity(FD[i], 500, 0, 0)
	print(mean_intensity)

	# Create function which calculates and plots the derivatives Ix and Iy

	# Create function which calculates and plots the square of teh derivatives Ix^2
	# Iy^2 and IxIy
	
	# Create a function which applies a gaussian filter to these images above
	
	# Create a cornerness function (see slide 48 of mlcv_featuredetection)

for i in range(0,1):
	mean_intensity, image_intensity, WSplits, HSplits = findIntensity(FD[i], 500, 0, 0)
	print(mean_intensity)

	# Create function which calculates and plots the derivatives Ix and Iy

	# Create function which calculates and plots the square of teh derivatives Ix^2
	# Iy^2 and IxIy
	
	# Create a function which applies a gaussian filter to these images above
	
	# Create a cornerness function (see slide 48 of mlcv_featuredetection)
