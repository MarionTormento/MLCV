def findHomography(Image1, Image2, ImageA, ImageB):

	#ImageA = img1.jpg
	#ImageB = img3.jpg
	#Set of points manually selected from images, should be replaced with our own
	#interest points
	# ImageA = np.array([[253, 183], 
	#                   [306, 196],
	#                   [397, 211],
	#                   [389, 329],
	#                   [473, 391],
	#                   [481, 279]])
	# ImageB = np.array([[287, 196],
	#                   [314, 222],
	#                   [359, 260],
	#                   [331, 359],
	#                   [362, 429], 
	#                   [387,338]])

	img1 = cv2.imread('Photos/' + Image1)
	img2 = cv2.imread('Photos/' + Image2)

	#set length of P matrix
	nbPoints = len(ImageA)
	P = np.zeros((2*nbPoints + 1, 9))

	#populate P matrix
	for i in range(1,nbPoints+1):
		P[(2*i-2):(2*i)][:] = np.array([[-ImageA[i-1][0], -ImageA[i-1][1], -1,             0,             0,  0, ImageA[i-1][0]*ImageB[i-1][0], ImageA[i-1][1]*ImageB[i-1][0] , ImageB[i-1][0]],
		                      			 [             0,             0,  0, -ImageA[i-1][0], -ImageA[i-1][1], -1, ImageA[i-1][0]*ImageB[i-1][1], ImageA[i-1][1]*ImageB[i-1][1] , ImageB[i-1][1]]])
	P[2*len(ImageA)][:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

	#Perform SVD
	U, S, VT = np.linalg.svd(P)

	# Set H as the last column of V (last row of VT) as it will cause least error
	H = np.zeros((3,3))
	H=VT[-1,:].reshape((3,3))

	# Find and print a test point to check it's working
	point_estimated_prime = np.dot(H, np.concatenate((ImageA, np.ones((nbPoints,1))), axis = 1).T)
	points_estimated = (point_estimated_prime[:][0:2] / point_estimated_prime[:][-1]).T
	print(points_estimated)
	print(ImageB)

	dist_diff = np.linalg.norm(ImageB-points_estimated, axis = 1)
	Homography_accuracy = np.mean(dist_diff)
	print(Homography_accuracy)




