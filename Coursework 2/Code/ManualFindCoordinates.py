import tkinter as tk
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import ImageTk, Image

def mouse_click(event):
    # retrieve XY coords as a tuple
    x = event.x
    y = event.y
    coords = (x,y)
    # coords = root.winfo_pointerxy()
    print('coords: {}'.format(coords))
    print('X: {}'.format(coords[0]))
    print('Y: {}'.format(coords[1]))
    file.write('{}, {} \n'.format(x, y))

Test_images = (['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'])
Quick1 = (['chess.png', 'chess2.png', 'chess3.png'])
Quick2 = (['chess.png', 'chess.jpg'])
test = Test_images

for i in range(2):

	print("New image")
	image = test[i]
	file = open("Image" + str(i) + ".csv", 'w')

	path = 'Photos/' + image
	root = tk.Tk()
	img = Image.open(path).convert('LA')
	# img = img.resize((600, 600), Image.ANTIALIAS)
	img = ImageTk.PhotoImage(img)
	panel = tk.Label(root, image = img)
	panel.pack(side = "bottom", fill = "both", expand = "yes")
	root.bind('<Button>', mouse_click)
	click = root.bind('<Button>', mouse_click)
	root.mainloop()

	file.close() 