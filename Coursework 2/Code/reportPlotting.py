from functionsCW2 import *

# ------------------------- Main Script --------------------------------

# import images
FD = (['fd1.jpg', 'fd2.jpg', 'fd3.jpg', 'fd4.jpg', 'fd5.jpg', 'fd6.jpg',
      'fd7.jpg', 'fd8.jpg', 'fd9.jpg', 'fd10.jpg', 'fd11.jpg', 'fd12.jpg', 'fd13.jpg'])

HD = (['3_2_1.jpg', '3_2_2.jpg', '3_2_3.jpg',  '4_0_1.jpg', '4_0_2.jpg',
      '4_0_3.jpg', '5_0_1.jpg', '5_0_2.jpg', '5_0_3.jpg'])

Tsukuba = (['Tsukuba1.jpg', 'Tsukuba2.jpg', 'Tsukuba3.jpg', 'Tsukuba4.jpg', 'Tsukuba5.jpg'])

Art = (['Art1.png', 'Art2.png', 'Art3.png', 'Art4.png', 'Art5.png', 'Art6.png', 'Art7.png'])

NakedMan = (['img_1360.jpg', 'img_1361.jpg', 'img_1362.jpg'])

Test_images = (['img1.jpg','img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', 'img0.jpg'])
#
Quick1 = (['chess.png', 'chess2.png', 'chess3.png'])
Quick2 = (['chess.png', 'chess.jpg'])
JBL = (['jbl1.jpg','jbl2.jpg','jbl3.jpg','jbl4.jpg'])
Map = (['map1.jpg','map2.jpg','map3.jpg','map4.jpg'])
MapRot = (['RotMap2.jpg', 'RotMap3.jpg', 'RotMap1.jpg', 'RotMap4.jpg'])

compRoom = (['comproom.jpg', 'comproom1.jpg'])

livingRoom = (['LivingRoom1.jpg', 'LivingRoom2.jpg'])

findPoints = 'Auto' #'Auto' or 'Manual' 
descriptorType = 'RGB' #'RGB' or 'HOG' or 'RGBHOG'
cornerDetectionType = 'FAST' #'FAST' or 'Harris' or 'ST'
ImplementedOrToolBox = 'Implemented' #'Implemented' or 'ToolBox'
allIntensity = []
allPoints = []
allDesc = []
test = MapRot

#FAST Parameters
FAST_radius = 3
FAST_S = 8
FAST_threshold = 45

#Harris/Shi-Tomasi Parameters
alpha = 0.04
Maxima_NN = 50 # Number of Nearest Neighbour
Maxima_perc = 99 # Percentage of value kept by the thresholding

# Gerenal Parameters
windowSize = 21 #WARNING : Must be uneven
derivative = 'No' # 'Yes' or 'No'

for i in [0]:

	print("New image")
	image = test[i]

	allIntensity = []
	allPoints = []
	allDesc = []

	intensity, imgPlot, shift = getImageIntensity(image)
	Ix, Iy = derivatives(intensity, shift, 0)
	sigma = 1.6*shift
	GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)

	ImplementedOrToolBox = 'Implemented'

	cornerDetectionType = 'Harris'
	CornerPointsHarris = cornerness_funct(intensity, imgPlot, GIxx, GIyy, GIxy, shift, alpha, windowSize, 0, Maxima_NN, Maxima_perc, cornerDetectionType)

	cornerDetectionType = 'ST'
	CornerPointsST = cornerness_funct(intensity, imgPlot, GIxx, GIyy, GIxy, shift, alpha, windowSize, 0, Maxima_NN, Maxima_perc, cornerDetectionType)

	cornerDetectionType = 'FAST'
	CornerPointsFAST = 	FASTdetector(intensity, imgPlot, FAST_radius, FAST_S, FAST_threshold)
	CornerPointsFAST = cleanSides(intensity, CornerPointsFAST, windowSize)
	CornerPointsFAST = (CornerPointsFAST[0], CornerPointsFAST[1])

	plt.figure()
	plt.title("Implemented Corner Detector Comparison")
	plt.imshow(imgPlot)
	Harris = plt.scatter(CornerPointsHarris[0], CornerPointsHarris[1], color='r', marker='o', facecolors='none', label="Harris")
	ST = plt.scatter(CornerPointsST[0], CornerPointsST[1], color='g', marker='^', facecolors='none', label="Shi-Tomasi")
	FAST = plt.scatter(CornerPointsFAST[0], CornerPointsFAST[1], color='b', marker='s', facecolors='none', label="FAST")
	plt.legend(handles=[Harris, ST, FAST], loc=4)

	ImplementedOrToolBox = 'ToolBox'

	cornerDetectionType = 'Harris'
	CornerPointsHarrisTB = CornerTB(image, cornerDetectionType, alpha, FAST_threshold)
	CornerPointsHarrisTB = cleanSides(intensity, CornerPointsHarrisTB, windowSize)
	CornerPointsHarrisTB = (CornerPointsHarrisTB[1], CornerPointsHarrisTB[0])

	cornerDetectionType = 'ST'
	CornerPointsSTTB = CornerTB(image, cornerDetectionType, alpha, FAST_threshold)
	CornerPointsSTTB = cleanSides(intensity, CornerPointsSTTB, windowSize)
	CornerPointsSTTB = (CornerPointsSTTB[1], CornerPointsSTTB[0])

	cornerDetectionType = 'FAST'
	CornerPointsFASTTB = CornerTB(image, cornerDetectionType, alpha, FAST_threshold)
	CornerPointsFASTTB = cleanSides(intensity, CornerPointsFASTTB, windowSize)
	CornerPointsFASTTB = (CornerPointsFASTTB[1], CornerPointsFASTTB[0])

	# plt.figure()
	# plt.suptitle("Inbuilt vs Implemented Corner Detector Comparison")

	#ax1 = plt.subplot(131)
	plt.figure()
	plt.title("Harris Corner Detectors")
	plt.imshow(imgPlot)
	Implemented = plt.scatter(CornerPointsHarris[0], CornerPointsHarris[1], color='r', marker='s', facecolors='none', label="Implemented")
	ToolBox = plt.scatter(CornerPointsHarrisTB[0], CornerPointsHarrisTB[1], color='g', marker='^', facecolors='none', label="Toolbox")
	plt.legend(handles=[Implemented,ToolBox], loc=4)

	#ax2 = plt.subplot(132)
	plt.figure()
	plt.title("Shi-Tomasi Corner Detectors")
	plt.imshow(imgPlot)
	Implemented = plt.scatter(CornerPointsST[0], CornerPointsST[1], color='r', marker='s', facecolors='none', label="Implemented")
	ToolBox = plt.scatter(CornerPointsSTTB[0], CornerPointsSTTB[1], color='g', marker='^', facecolors='none', label="Toolbox")
	plt.legend(handles=[Implemented,ToolBox], loc=4)

	#ax3 = plt.subplot(133)
	plt.figure()
	plt.title("Fast Corner Detectors")
	plt.imshow(imgPlot)
	Implemented = plt.scatter(CornerPointsFAST[0], CornerPointsFAST[1], color='r', marker='s', facecolors='none', label="Implemented")
	ToolBox = plt.scatter(CornerPointsFASTTB[0], CornerPointsFASTTB[1], color='g', marker='^', facecolors='none', label="Toolbox")
	plt.legend(handles=[Implemented,ToolBox], loc=4)

	plt.show()

