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
Map = (['map1.jpg','mabp2.jpg','map3.jpg','map4.jpg'])
MapRot = (['RotMap2.jpg', 'RotMap3.jpg', 'RotMap1.jpg', 'RotMap4.jpg'])

compRoom = (['comproom.jpg', 'comproom1.jpg'])

livingRoom = (['LivingRoom1.jpg', 'LivingRoom2.jpg'])

final = (['Final1.jpg', 'Final2.jpg'])

findPoints = 'Auto' #'Auto' or 'Manual' 
descriptorType = 'RGB' #'RGB' or 'HOG' or 'RGBHOG'
cornerDetectionType = 'Harris' #'FAST' or 'Harris' or 'ST'
ImplementedOrToolBox = 'Implemented' #'Implemented' or 'ToolBox'
allIntensity = []
allPoints = []
allDesc = []
test = Tsukuba

#FAST Parameters
FAST_radius = 3
FAST_S = 9
FAST_threshold = 30

#Harris/Shi-Tomasi Parameters
alpha = 0.04
Maxima_NN = 50 # Number of Nearest Neighbour
Maxima_perc = 98 # Percentage of value kept by the thresholding

# Gerenal Parameters
windowSize = 21 #WARNING : Must be uneven
derivative = 'No' # 'Yes' or 'No'

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

R, T = rigid_transform_3D(corrBasePoints, corrTestPoints)

print('Rotation = ')
print(R)
scale2 = np.linalg.det(R[0:2,0:2])
print(scale2, np.sqrt(scale2))
rotAngle = 180*acos(R[0,0])/np.pi
print(rotAngle)

print('Translation = ')
print(T)

ImageAgood, ImageBgood, H, acc_homog, acc_homog_norm, im_rec, im_rec_points = findHomography(test[0], test[1], corrBasePoints, corrTestPoints, 4)

T = np.asarray([T])
T = T.T
# R = np.eye(3)
f = 1
K = np.array([[f, 0, ]])

# stereoRectification(test[0], test[1], corrBasePoints, corrTestPoints, T, R, f)

disparityMap, depthMap = dispMap(test[0], im_rec, 5, derivative)
# disparityMap = cv2.applyColorMap(disparityMap, cv2.COLORMAP_JET)
plt.figure(6)
plt.subplot(121), plt.imshow(disparityMap, interpolation='nearest', cmap='gray')
plt.subplot(122), plt.imshow(depthMap, interpolation='nearest', cmap='gray')#, norm=NoNorm())
# ax = sns.heatmap(disparityMap, linewidth=0.5)
plt.colorbar(orientation='horizontal')

X,Y = np.meshgrid(np.arange(depthMap.shape[1]), np.arange(depthMap.shape[0]))

fig = plt.figure(15)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,depthMap, cmap='gray')# vmin = np.amin(depthMap), vmax = np.amax(depthMap))
fig2 = plt.figure(16)
ax2 = fig2.gca(projection='3d')
scatter = ax2.scatter(X, Y, depthMap, cmap = 'gray')

acc_fund, acc_fund_norm = findFundamental(test[0], im_rec, corrBasePoints, corrTestPoints)

print('Homography Accuracy = %1.2f' % acc_homog)
print('Normalised Homography Accuracy = %1.2f' % acc_homog_norm)
print('Fundamental Accuracy = %1.2f' % acc_fund)
print('Normalised Fundamental Accuracy = %1.2f' % acc_fund_norm)

acc_fund = findFundamental(test[0], test[1], corrBasePoints, im_rec_points)

plt.show()
