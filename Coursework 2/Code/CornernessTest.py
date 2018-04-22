from functionsCW2 import *
from math import atan2
from itertools import groupby

def getConsec(arr):
 
	count = 0
	result = 0

	for i in range(0, len(arr)):
		if (arr[i] == 0):
			count = 0
		else:
			count+= 1
			result = max(result, count) 
		 
	return result 

def get_circle(centre, radius):

	x0 = centre[0]
	y0 = centre[1]

	x, y, p = 0, radius, 1-radius

	L = []
	L.append((x, y))

	for x in range(int(radius)):
		if p < 0:
			p = p + 2 * x + 3
		else:
			y -= 1
			p = p + 2 * x + 3 - 2 * y

		L.append((x, y))

		if x >= y: break

	N = L[:]
	for i in L:
		N.append((i[1], i[0]))

	L = N[:]
	for i in N:
		L.append((-i[0], i[1]))
		L.append((i[0], -i[1]))
		L.append((-i[0], -i[1]))

	N = []
	for i in L:
		N.append((x0+i[0], y0+i[1]))

	return N

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

findPoints = 'Auto' #'Auto' or 'Manual'
alpha = 0.02 
descriptorType = 'RGB' #'RGB' or 'HOG'
ImplementedOrToolBox = 'Implemented' #'Implemented' or 'ToolBox'
allIntensity = []
allPoints = []
allDesc = []
test = Test_images
windowSize = 21 #WARNING : Must be uneven


radius = 3
S = 6
threshold = 40

print("New image")
image = test[0]
image, shift = getImageIntensity(image)
width, height = image.shape

plt.imshow(image)
print(image)

wide = np.arange(radius,width-radius-1)
high = np.arange(radius,height-radius-1)

cornerPoints = []

for i in wide[::2]:
	for j in high[::2]:
		pixelI = image[i][j]
		N = get_circle([i,j],radius)
		N = list(set(N))
		N = np.asarray(N)
		angles = []
		for n in range(0,len(N)):
			angle = atan2(N[n,0] - i, N[n,1] - j)
			angles.append(angle)
		sortedIdx = np.argsort(angles)
		N = N[sortedIdx]
		zN = list(zip(*N))
		CircleInt = image[zN]
		greaterThan = CircleInt > pixelI + threshold
		lessThan = CircleInt < pixelI - threshold
		greaterThan = greaterThan.astype(int)
		GTmatrix = np.concatenate((greaterThan,greaterThan), axis=0)
		consecGT = getConsec(GTmatrix)
		lessThan = lessThan.astype(int)
		LTmatrix = np.concatenate((lessThan,lessThan), axis=0)
		consecLT = getConsec(LTmatrix)
		if consecGT > S or consecLT > S:
			cornerPoints.append([i, j])

cornerPoints = np.asarray(cornerPoints)
cornerPointsX = cornerPoints[:][:,1]
cornerPointsY = cornerPoints[:][:,0]
plt.scatter(cornerPointsX, cornerPointsY, marker='+', color='red')
plt.show()