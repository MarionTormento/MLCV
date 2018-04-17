
from functionsCW2 import *

# ------------------------- Main Script --------------------------------

# import images
FD = (['FD1.jpg', 'FD2.jpg', 'FD3.jpg', 'FD4.jpg', 'FD5.jpg', 'FD6.jpg',
      'FD7.jpg', 'FD8.jpg', 'FD9.jpg', 'FD10.jpg', 'FD11.jpg', 'FD12.jpg', 'FD13.jpg'])

HD = (['3.2_1.jpg', '3.2_2.jpg', '3.2_3.jpg',  '4.0_1.jpg', '4.0_2.jpg',
      '4.0_3.jpg', '5.0_1.jpg', '5.0_2.jpg', '5.0_3.jpg'])

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])

Quick1 = (['chess.png', 'chess2.png', 'chess3.png'])
Quick2 = (['chess.png', 'chess.jpg'])

allIntensity = []
allPoints = []
allDesc = []
test = Test_images
windowSize = 21 #WARNING : Must be uneve


for i in range(2):

	print("New image")
	image = test[i]
	intensity, shift = getImageIntensity(image)

	# print("Manually find interest points")
	# root = tk.Tk()
	# img = Image.open('Photos/' + image).convert('LA')
	# img = ImageTk.PhotoImage(img)
	# panel = tk.Label(root, image = img)
	# panel.pack(side = "bottom", fill = "both", expand = "yes")
	# root.bind('<Button>', mouse_click)
	# root.mainloop()

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
u = knn("color", allIntensity, allDesc, allPoints, 0, 1, 1)


