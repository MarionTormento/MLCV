""" Where's My Mouse? """
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

path = 'Photos/img3.JPG'
root = tk.Tk()
img = Image.open(path).convert('LA')
# img = img.resize((600, 600), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
panel = tk.Label(root, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
root.bind('<Button>', mouse_click)
root.mainloop()