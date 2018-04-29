from PIL import Image

RotMap =(['RotMap2.jpg','RotMap1.jpg','RotMap3.jpg','RotMap4.jpg','RotMap5.jpg'])

img = Image.open('img1.jpg')
width, height = img.size[0], img.size[1]
new_img = img.resize((int(width/3),int(height/3)))
new_img.save("img1_3.jpg", "JPEG")