from functionsCW2 import *

# ------------------------- Main Script --------------------------------

# import images
FD = (['fd1.jpg', 'fd2.jpg', 'fd3.jpg', 'fd4.jpg', 'fd5.jpg', 'fd6.jpg',
      'fd7.jpg', 'fd8.jpg', 'fd9.jpg', 'fd10.jpg', 'fd11.jpg', 'fd12.jpg', 'fd13.jpg'])

HD = (['3_2_1.jpg', '3_2_2.jpg', '3_2_3.jpg',  '4_0_1.jpg', '4_0_2.jpg',
      '4_0_3.jpg', '5_0_1.jpg', '5_0_2.jpg', '5_0_3.jpg'])

NakedMan = (['IMG_1362.jpg', 'IMG_1361.jpg', 'IMG_1360.jpg'])

Test_images = (['img1.jpg','img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg', 'img0.jpg'])

Quick1 = (['chess.png', 'chess2.png', 'chess3.png'])
Quick2 = (['chess.png', 'chess.jpg'])

findPoints = 'Auto'
allIntensity = []
allPoints = []
allDesc = []
test = Test_images
windowSize = 21 #WARNING : Must be uneven


for i in range(2):

	print("New image")
	image = test[i]
	intensity, shift = getImageIntensity(image)
	
	if findPoints == 'Manual':

		CornerPoints = manualCornerPoints(image, i)

	elif findPoints == 'Auto':

		print("Automatically find interest Points")
		print("Computing Intensity derivatives")
		Ix, Iy = derivatives(intensity, shift, 0)
		sigma = 1.6*shift
		GIxx, GIyy, GIxy = gaussian_window(Ix, Iy, sigma, shift)

		print("Identifying corners and edges")
		CornerPoints = cornerness_funct(intensity, GIxx, GIyy, GIxy, shift, 0.05, windowSize, 0)

	print("Computing RGB descriptor")
	desc = descripter_funct(CornerPoints, image, windowSize, 0)
	
	# print("Computing histogram of gradient orientation")
	# desc = hog(intensity, Ix, Iy, CornerPoints, windowSize, 0)

	print("Saving all values")
	allDesc.append(desc)
	allIntensity.append(intensity)
	allPoints.append(CornerPoints)

print("Looking for matching descriptors")
indexNN, corrBasePoints, corrTestPoints = knn("color", allIntensity, allDesc, allPoints, 0, 1, 1)

ImageAgood, ImageBgood, acc_homog = findHomography(test[0], test[1], corrBasePoints, corrTestPoints, 4)

acc_fund = findFundamental(test[0], test[1], ImageAgood, ImageBgood)

print('Homography Accuracy = %1.2f' % acc_homog)
print('Fundamental Accuracy = %1.2f' % acc_fund)

plt.show()
