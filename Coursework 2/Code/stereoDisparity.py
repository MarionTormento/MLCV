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
from math import atan2
from itertools import groupby
from functionsCW2 import *

def dispMap(Image1, Image2):

	shift = 8
	# Load images in grayscale
	img1 = cv2.imread('Photos/' + Image1,0)
	img1 = np.asarray(img1)
	Ix1, Iy1 = derivatives(img1, shift, 0)
	img1 = Iy1*Ix1

	img2 = cv2.imread('Photos/' + Image2,0)
	img2 = np.asarray(img2)
	Ix2, Iy2 = derivatives(img2, shift, 0)
	img2 = Iy2*Ix2

	# Initialisation
	windowSize = int(min(img1.shape)/20)
	if np.mod(windowSize,2) == 0:
		windowSize += 1
	print(windowSize)
	halfWS = int((windowSize-1)/2)
	disparityMap = np.zeros(img1.shape)
	height, width = img1.shape
	disparityRange = 50 #int(min(width, height)/10)

	# Looping
	for i in range(height):
		minH = max(0, i-halfWS)
		maxH = min(height, i+halfWS)
		for j in range(width):
			minW = max(0, j-halfWS)
			maxW = min(width, j+halfWS)
			minD = max(-disparityRange, -minW)
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
	return disparityMap

disp = dispMap('left.png', 'right.png')
# resultat = resultat.astype("float")
print(disp)
# print(depth)
plt.figure()
plt.subplot(211), plt.imshow(disp, interpolation='nearest', cmap='gray')
# plt.subplot(212), plt.imshow(depth, interpolation='nearest', cmap='gray')
plt.show()