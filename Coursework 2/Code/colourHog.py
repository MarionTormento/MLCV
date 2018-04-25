from functionsCW2 import *

Art = (['Art1.png', 'Art2.png', 'Art3.png', 'Art4.png', 'Art5.png', 'Art6.png', 'Art7.png'])

allIntensity = []
allPoints = []
allDesc = []
test = Art
windowSize = 21

for i in [0,1]:

	print("New image")
	image = test[i]

	intensity, shift = getImageIntensity(image)

	print("Computing 'FAST' Corner Detector")
	radius = 3
	S = 8
	threshold = 50
	CornerPoints = FASTdetector(intensity, radius, S, threshold)
	imagedesc = np.zeros((len(CornerPoints[0]), 108))
	img = cv2.imread('Photos/' + image)
	red = img[:,:,2]
	green = img[:,:,1]
	blue = img[:,:,0]
	count = 0

	for j in [blue, green, red]:

		print("Computing histogram of gradient orientation")
		Ix, Iy = derivatives(j, shift, 0)
		desc = hog(j, Ix, Iy, CornerPoints, windowSize, 0)
		imagedesc[:,36*count:36*(count+1)] = desc
		count += 1

	imagedesc = np.asarray(imagedesc)
	allDesc.append(imagedesc)
