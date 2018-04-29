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
test = final 

#FAST Parameters
FAST_radius = 3
FAST_S = 9
FAST_threshold = 60

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

<<<<<<< HEAD
disparityMap, depthMap, depthMap1 = dispMap(test[1], test[0], 7, derivative, T)
# disparityMap = cv2.applyColorMap(disparityMap, cv2.COLORMAP_JET)

# plt.figure(10)
# stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
# disparity = stereo.compute(allIntensity[0], allIntensity[1])
# plt.imshow(disparity,'gray')
# plt.show()

# Define the gaussian window of size shift x shift
shift = 5
sigma = 1.6*shift
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
disparityGauss = signal.convolve2d(disparityMap, gauss, 'same')
width, height = disparityGauss.shape
disparityMapMin = np.amin(disparityGauss)
disparityMapMax = np.amax(disparityGauss)
depthGauss = np.zeros(disparityGauss.shape)

for i in range(height):
	for j in range(width):
		# disparityGauss[j,i] = - disparityMapMin + disparityGauss[j,i]*3/(disparityMapMax - disparityMapMin)
		depthGauss[j,i] = 20/disparityGauss[j,i]

disparityMapMin = np.amin(disparityMap)
disparityMapMax = np.amax(disparityMap)
bounds = range(disparityMapMin, disparityMapMax, (disparityMapMax-disparityMapMin)//5)
disparityMapMin1 = np.amin(depthMap)
disparityMapMax1 = np.amax(depthMap)
bounds1 = range(disparityMapMin1, disparityMapMax1, (disparityMapMax1-disparityMapMin1)//5)
disparityMapMin3 = np.amin(depthGauss)
disparityMapMax3 = np.amax(depthGauss)
bounds3 = range(disparityMapMin3, disparityMapMax3, (disparityMapMax3-disparityMapMin3)//5)
disparityMapMin2 = np.amin(depthMap1)
disparityMapMax2 = np.amax(depthMap1)
bounds2 = range(disparityMapMin2, disparityMapMax2, (disparityMapMax2-disparityMapMin2)//5)

fig11 = plt.figure()
plt.suptitle('Depth and Disparity', fontsize=12)

ax1 = plt.subplot(221)
ax1.set_title('Disparity Map', fontsize=12)
im1 = ax1.imshow(disparityMap, interpolation='nearest', cmap='gray')
CB1 = fig11.colorbar(im1, orientation='horizontal', ticks=bounds, spacing='uniform')
CB1.set_label('Distance (pixels)', fontsize=12)
plt.xlabel('Pixels', fontsize=12)
plt.ylabel('Pixels', fontsize=12)
ax2 = plt.subplot(222)
ax2.set_title('Depth Map', fontsize=12)
im2 = ax2.imshow(depthMap, interpolation='nearest', cmap='gray', norm=NoNorm())
CB2 = fig11.colorbar(im2, orientation='horizontal', ticks=bounds1, spacing='uniform')
CB2.set_label('Distance (cm)', fontsize=12)
plt.xlabel('Pixels', fontsize=12)
plt.ylabel('Pixels', fontsize=12)
ax3 = plt.subplot(223)
ax3.set_title('Disparity Map (Focal Increase)', fontsize=12)
im3 = ax3.imshow(depthMap1, interpolation='nearest', cmap='gray', norm=NoNorm())
CB3 = fig11.colorbar(im3, orientation='horizontal', ticks=bounds2, spacing='uniform')
plt.xlabel('Pixels', fontsize=12)
plt.ylabel('Pixels', fontsize=12)
ax4 = plt.subplot(224)
ax4.set_title('Depth Map (Noisey)', fontsize=12)
im4 = ax4.imshow(depthGauss, interpolation='nearest', cmap='gray', norm=NoNorm())
CB4 = fig11.colorbar(im4, orientation='horizontal', ticks=bounds3, spacing='uniform')
CB4.set_label('Distance (cm)', fontsize=12)
plt.xlabel('Pixels', fontsize=12)
plt.ylabel('Pixels', fontsize=12)

fig22 = plt.figure()
plt.suptitle('Disparity and depth ', fontsize=12)

ax3 = plt.subplot(121)
ax3.set_title('Disparity Map (Noisey)', fontsize=12)
im3 = ax3.imshow(disparityGauss, interpolation='nearest', cmap='gray')
fig22.colorbar(im3, orientation='horizontal')
plt.xlabel('Pixels', fontsize=12)
plt.ylabel('Pixels', fontsize=12)
ax4 = plt.subplot(122)
ax4.set_title('Depth Map (Noisey)', fontsize=12)
im4 = ax4.imshow(depthGauss, interpolation='nearest', cmap='gray', norm=NoNorm())
fig22.colorbar(im4, orientation='horizontal')
plt.xlabel('Pixels', fontsize=12)
plt.ylabel('Pixels', fontsize=12)

X,Y = np.meshgrid(np.arange(depthMap.shape[1]), np.arange(depthMap.shape[0]))

fig = plt.figure(15)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,depthMap, cmap='gray')# vmin = np.amin(depthMap), vmax = np.amax(depthMap))
# fig2 = plt.figure(16)
# ax2 = fig2.gca(projection='3d')
# scatter = ax2.scatter(X, Y, depthMap, cmap = 'gray')

acc_fund, acc_fund_norm = findFundamental(test[0], test[1], corrBasePoints, corrTestPoints)

print('Homography Accuracy = %1.2f' % acc_homog)
print('Normalised Homography Accuracy = %1.2f' % acc_homog_norm)
print('Fundamental Accuracy = %1.2f' % acc_fund)
print('Normalised Fundamental Accuracy = %1.2f' % acc_fund_norm)

acc_fund_rec, acc_fund_norm_rec = findFundamental(test[0], im_rec, corrBasePoints, im_rec_points)

print('Im Rec Fundamental Accuracy = %1.2f' % acc_fund_rec)
print('Im Rec Normalised Fundamental Accuracy = %1.2f' % acc_fund_norm_rec)

plt.show()
