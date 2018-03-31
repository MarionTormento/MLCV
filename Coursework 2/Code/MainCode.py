import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def findIntensity(image, inputSplit, plot1, plot2):
    # Load an color image in grayscale
    img = cv2.imread('Photos/' + image, 0)
    width, height = img.shape

    if plot1>0:
        fig1 = plt.figure()
        plt.subplot(211), plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.subplot(212), plt.hist(img.ravel(),256,[0,256]);
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    # create a mask
    splits = int(inputSplit)

    for i in range(0,splits):
        for ii in range(0,splits):
            mask = np.zeros(img.shape[:2], np.uint8)
            mask[int((i)*width/splits):int((i+1)*width/splits), int((ii)*height/splits):int((ii+1)*height/splits)] = 255
            masked_img = cv2.bitwise_and(img,img,mask = mask)
            hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

            if plot2>0:
                figure = plt.figure()
                plt.subplot(211), plt.imshow(masked_img, 'gray')
                plt.subplot(212), plt.plot(hist_mask)
                plt.xlim([0,256])

            del hist_mask

    plt.show()

FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
'4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

for i in range(0,1):
    findIntensity(FD[i], 2, 0, 1)

# for i in range(0,len(HD)):
#     findIntensity(HD[i],2, 0)

