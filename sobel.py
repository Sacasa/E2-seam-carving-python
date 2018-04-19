import time
import sys
from PIL import Image,ImageFilter
import threading
import math

def sobel(im,cp,buffer,buffer2,mat_x,mat_y):
	for x in range(im.size[0]):
		for y in range(im.size[1]):
			gx=0
			gy=0
			for x_p in range(x-1,x+2):
				for y_p in range(y-1,y+2):
					if (0<=x_p<im.size[0]) and (0<=y_p<im.size[1]):
						color = buffer[x_p,y_p]
						coeff_x = mat_x[x_p-x+1][y_p-y+1]
						coeff_y = mat_y[x_p-x+1][y_p-y+1]
						gx+=color*coeff_x
						gy+=color*coeff_y
			res = (gx*gx)+(gy*gy)
			buffer2[x,y]=int(math.sqrt(res))
	return cp

def sobel1(im,cp,buffer,buffer2,mat_x,mat_y):
	for x in range(im.size[0]):
		for y in range(im.size[1]//4):
			gx=0
			gy=0
			for x_p in range(x-1,x+2):
				for y_p in range(y-1,y+2):
					if (0<=x_p<im.size[0]) and (0<=y_p<im.size[1]):
						color = buffer[x_p,y_p]
						coeff_x = mat_x[x_p-x+1][y_p-y+1]
						coeff_y = mat_y[x_p-x+1][y_p-y+1]
						gx+=color*coeff_x
						gy+=color*coeff_y
			res = (gx*gx)+(gy*gy)
			buffer2[x,y]=int(math.sqrt(res))

def sobel2(im,cp,buffer,buffer2,mat_x,mat_y):
	for x in range(im.size[0]):
		for y in range(im.size[1]//4,im.size[1]//2):
			gx=0
			gy=0
			for x_p in range(x-1,x+2):
				for y_p in range(y-1,y+2):
					if (0<=x_p<im.size[0]) and (0<=y_p<im.size[1]):
						color = buffer[x_p,y_p]
						coeff_x = mat_x[x_p-x+1][y_p-y+1]
						coeff_y = mat_y[x_p-x+1][y_p-y+1]
						gx+=color*coeff_x
						gy+=color*coeff_y
			res = (gx*gx)+(gy*gy)
			buffer2[x,y]=int(math.sqrt(res))

def sobel3(im,cp,buffer,buffer2,mat_x,mat_y):
	for x in range(im.size[0]):
		for y in range(im.size[1]//4,3*(im.size[1]//2)//4):
			gx=0
			gy=0
			for x_p in range(x-1,x+2):
				for y_p in range(y-1,y+2):
					if (0<=x_p<im.size[0]) and (0<=y_p<im.size[1]):
						color = buffer[x_p,y_p]
						coeff_x = mat_x[x_p-x+1][y_p-y+1]
						coeff_y = mat_y[x_p-x+1][y_p-y+1]
						gx+=color*coeff_x
						gy+=color*coeff_y
			res = (gx*gx)+(gy*gy)
			buffer2[x,y]=int(math.sqrt(res))

def sobel4(im,cp,buffer,buffer2,mat_x,mat_y):
	for x in range(im.size[0]):
		for y in range(3*(im.size[1]//2)//4,im.size[1]):
			gx=0
			gy=0
			for x_p in range(x-1,x+2):
				for y_p in range(y-1,y+2):
					if (0<=x_p<im.size[0]) and (0<=y_p<im.size[1]):
						color = buffer[x_p,y_p]
						coeff_x = mat_x[x_p-x+1][y_p-y+1]
						coeff_y = mat_y[x_p-x+1][y_p-y+1]
						gx+=color*coeff_x
						gy+=color*coeff_y
			res = (gx*gx)+(gy*gy)
			buffer2[x,y]=int(math.sqrt(res))

def Main():
	im=Image.open("Capture.PNG")
	im = im.convert("L")
	buffer = im.load()
	cp = im.copy()
	buffer2=cp.load()
	mat_x= [[-1,0,1],[-2,0,2],[-1,0,1]]
	mat_y= [[-1,-2,-1],[0,0,0],[1,2,1]]

	p1 = threading.Thread(target=sobel1, args=(im,cp,buffer,buffer2,mat_x,mat_y,))
	p2 = threading.Thread(target=sobel2, args=(im,cp,buffer,buffer2,mat_x,mat_y,))
	p3 = threading.Thread(target=sobel3, args=(im,cp,buffer,buffer2,mat_x,mat_y,))
	p4 = threading.Thread(target=sobel4, args=(im,cp,buffer,buffer2,mat_x,mat_y,))

	tmps1=time.time()

	p1.start()
	p2.start()
	p3.start()
	p4.start()

	p1.join()
	p2.join()
	p3.join()
	p4.join()

	tmps2=time.time()

	tmps3=time.time()

	cp2 = sobel(im,cp,buffer,buffer2,mat_x,mat_y)

	tmps4=time.time()

	print("Temps calcul avec multithreading :", tmps2-tmps1)
	print("Temps calcul sans multithreading :", tmps4-tmps3)
	cp2.show()
	cp.show()

if __name__ == '__main__':
	Main()