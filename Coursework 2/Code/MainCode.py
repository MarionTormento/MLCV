from functionsCW2 import *

# ------------------------- Main Script --------------------------------

# import images
FD = (['fd1.jpg', 'fd2.jpg', 'fd3.jpg', 'fd4.jpg', 'fd5.jpg', 'fd6.jpg',
      'fd7.jpg', 'fd8.jpg', 'fd9.jpg', 'fd10.jpg', 'fd11.jpg', 'fd12.jpg', 'fd13.jpg'])

HD = (['3_2_1.jpg', '3_2_2.jpg', '3_2_3.jpg',  '4_0_1.jpg', '4_0_2.jpg',
      '4_0_3.jpg', '5_0_1.jpg', '5_0_2.jpg', '5_0_3.jpg'])

Tsukuba = (['Tsukuba1.jpg', 'Tsukuba2.jpg', 'Tsukuba3.jpg', 'Tsukuba4.jpg', 'Tsukuba5.jpg'])

Art = (['Art1.png', 'Art2.png', 'Art3.png', 'Art4.png', 'Art5.png', 'Art6.png', 'Art7.png'])

NakedMan = (['IMG_1362.jpg', 'IMG_1361.jpg', 'IMG_1360.jpg'])

Test_images = (['img1.jpg','img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', 'img0.jpg'])

Quick1 = (['chess.png', 'chess2.png', 'chess3.png'])
Quick2 = (['chess.png', 'chess.jpg'])
JBL = (['JBL1.jpg','JBL2.jpg','JBL3.jpg','JBL4.jpg'])
Map = (['map1.jpg','map2.jpg','map3.jpg','map4.jpg'])


findPoints = 'Auto' #'Auto' or 'Manual' 
descriptorType = 'RGB' #'RGB' or 'HOG' or 'RGBHOG'
cornerDetectionType = 'Harris' #'FAST' or 'Harris' or 'ST'
ImplementedOrToolBox = 'ToolBox' #'Implemented' or 'ToolBox'
allIntensity = []
allPoints = []
allDesc = []
test = Map

#FAST Parameters
FAST_radius = 3
FAST_S = 9
FAST_threshold = 40

#Harris/Shi-Tomasi Parameters
alpha = 0.04
Maxima_NN = 50 # Number of Nearest Neighbour
Maxima_perc = 99 # Percentage of value kept by the thresholding

# Gerenal Parameters
windowSize = 21 #WARNING : Must be uneven

for i in [0,1]:

	print("New image")
	image = test[i]

	desc, intensity, CornerPoints = getCornerPoints(image, i, alpha, findPoints, ImplementedOrToolBox, cornerDetectionType, descriptorType, windowSize, FAST_S, FAST_radius, FAST_threshold,  Maxima_NN, Maxima_perc)

	print("Saving all values")
	allDesc.append(desc)
	allIntensity.append(intensity)
	allPoints.append(CornerPoints)

print("Looking for matching descriptors")
indexNN, corrBasePoints, corrTestPoints = knn(descriptorType, allIntensity, allDesc, allPoints, 0, 1, 1)

R, T = rigid_transform_3D(corrBasePoints, corrBasePoints)

print('Rotation = ')
print(R)

print('Translation = ')
print(T)

ImageAgood, ImageBgood, H, acc_homog, acc_homog_norm, im_rec = findHomography(test[0], test[1], corrBasePoints, corrTestPoints, 4)

# theta = 10*np.pi/180
# T0 = [10*np.cos(theta), 10*np.sin(theta), 0]
# T0 = np.asarray([T0])
# T0 = T0.T
# R = [[np.cos(theta), -np.sin(theta), 0],
# 	[np.sin(theta), np.cos(theta), 0],
# 	[0, 0, 1]]
# f = 20
# stereoRectification(test[0], test[1], corrBasePoints, corrTestPoints, T0, R, f)

# disparityMap = dispMap(test[0], im_rec, 7)
# # disparityMap = cv2.applyColorMap(disparityMap, cv2.COLORMAP_JET)
# plt.figure(6)
# plt.imshow(disparityMap, interpolation='nearest')
# # plt.colorbar()
 
acc_fund, acc_fund_norm = findFundamental(test[0], test[1], corrBasePoints, corrTestPoints)

print('Homography Accuracy = %1.2f' % acc_homog)
print('Normalised Homography Accuracy = %1.2f' % acc_homog_norm)
print('Fundamental Accuracy = %1.2f' % acc_fund)
print('Normalised Fundamental Accuracy = %1.2f' % acc_fund_norm)

acc_fund = findFundamental(test[0], im_rec, corrBasePoints, corrTestPoints)

plt.show()
