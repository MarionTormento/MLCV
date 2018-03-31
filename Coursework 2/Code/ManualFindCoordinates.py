""" Where's My Mouse? """
import tkinter as tk
import cv2
import numpy as np
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
    
root = tk.Tk()
img = Image.open('C:/Users/eddym/Documents/GitHub/MLCV1/Coursework 2/Code/Photos/3.2_1.JPG').convert('LA')
img = img.resize((600, 600), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)
panel = tk.Label(root, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
root.bind('<Button>', mouse_click)
root.mainloop()