import sys
from PIL import Image, ImageFilter

def calc_power_tab(im):
	buffer=im.load()
	power_tab=[[0]*im.size[1] for i in range(im.size[0])]
	for x in range(im.size[0]):
		power_tab[x][0]=buffer[x,0]
		for y in range(1,im.size[1]):
			power_last_min=300
			elem_power=0
			for x_p in range(x-1,x+2):
				if (0<=x_p<im.size[0]) and (0<=y-1<im.size[1]):
					elem_power=power_tab[x_p][y-1]
					if elem_power < power_last_min:
						power_last_min=elem_power
			power = power_last_min+buffer[x,y]
			power_tab[x][y]=power
	return power_tab

def find_power_path(power_tab,im):
	min_x=300
	path_tab=[[0]*im.size[1] for i in range(im.size[0])]
	for x in range(im.size[0]):
		if power_tab[x][0] < min_x:
			min_x = power_tab[x][0]

	for y in range(im.size[1]):
		for x_p in range(min_x-1,min_x+2):
			power_last_max=0
			elem_power=0
			if (0<=x_p<im.size[0]) and (0<=y+1<im.size[1]):
					elem_power=power_tab[x_p][y+1]
					if elem_power > power_last_max:
							power_last_max=elem_power
							min_x=x_p
		path_tab[min_x][y]=255
	return path_tab

def display_path_tab(path_tab,im):
	im=im.convert("RGB")
	buffer2=im.load()
	for x in range(im.size[0]):
		for y in range(im.size[1]):
			if path_tab[x][y]==255:
				buffer2[x,y]=(255,0,0)
	return im

def forward_im(im,path_tab):
	cp=Image.new("L",(im.size[0]-1,im.size[1]),0)
	im=im.convert("L")
	buffer2=cp.load()
	buffer=im.load()
	x_path = [0]*cp.size[1]
	for x in range(cp.size[0]):
		for y in range(cp.size[1]):
			if path_tab[x][y] == 255:
				x_path[y] = x
	for y in range(cp.size[1]):
		for x in range(x_path[y]):
			buffer2[x,y]=buffer[x,y]
		for x in range(x_path[y],cp.size[0]):
			if (1<=x<cp.size[0]):
				buffer2[x,y]=buffer[x+1,y]
	return cp

im=Image.open(sys.argv[1] +".jpg")
#im=im.resize((im.size[0]//4,im.size[1]//4))
im.show()
for i in range(100):
	truc = im.copy()
	im_bw=im.convert("L")
	cp=im_bw.copy()
	im_sobel = im_bw.filter(ImageFilter.FIND_EDGES)
	im=im_sobel
	power_tab=calc_power_tab(im)
	path_tab=find_power_path(power_tab,im)
	truc=display_path_tab(path_tab,truc)
	im=forward_im(cp,path_tab)
	print(i)
im.show()
im.save(sys.argv[1] + "_res.jpg", "JPEG")
		

