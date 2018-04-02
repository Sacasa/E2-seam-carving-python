from PIL import Image
import numpy as np
from numba import autojit



def dynamic_programming(mat):
    cop = np.zeros((len(mat),len(mat[0])) , dtype=np.int16)
    for y in range(len(mat)):
        for x in range(len(mat[0])):
            if y == 0:
                cop[y][x] = mat[y][x]
            elif x == 0:
                cop[y][x]= mat[y][x] + min([cop[y-1][x],cop[y-1][x+1]])
            elif x == len(mat[0])-1:
                cop[y][x]= mat[y][x] + min([cop[y-1][x],cop[y-1][x-1]])
            else:
                cop[y][x]= mat[y][x] + min([cop[y-1][x-1],cop[y-1][x],cop[y-1][x+1]])

    start = argmin(cop[-1])
    seam = np.array([[start,0]],dtype=np.int16)
    x = start
    dx = np.array([-1,0,1],dtype=np.int16)
    for y in range(len(cop)-1,0,-1):
        center = cop[y-1][x]
        if x == 0:
            values = np.array([32000,center,cop[y-1][x+1]],dtype=np.int16)
        elif x == len(cop[0])-1:
            values = np.array([cop[y-1][x-1],center,32000],dtype=np.int16)
        else:
            values = np.array([cop[y-1][x-1],center,cop[y-1][x+1]],dtype=np.int16)
        min_val = min(values)
        x += dx[argmin(values)]
        seam = np.append(seam,[[x,y-1]],axis=0)

    return seam

def move(im,list):
    cop = Image.new("RGB",(im.size[0]-1,im.size[1]),"black")
    pix = im.load()
    pix_cop = cop.load()
    for y in range(im.size[1]):
        for x in range(cop.size[0]):
            if x < list[y][0]:
                pix_cop[x,y] = pix[x,y]
            else:
                pix_cop[x,y] = pix[x+1,y]
    return cop

def move_l(im,list):
    cop = Image.new("L",(im.size[0]-1,im.size[1]))
    pix = im.load()
    pix_cop = cop.load()
    for y in range(im.size[1]):
        for x in range(cop.size[0]):
            if x < list[y][0]:
                pix_cop[x,y] = pix[x,y]
            else:
                pix_cop[x,y] = pix[x+1,y]
    return cop

@autojit
def move_mat(im,list):
    cop = np.zeros((len(im),len(im[0])-1), dtype=np.int16)
    for y in range(len(im)):
        for x in range(len(cop[0])):
            if x < list[y][0]:
                cop[y][x] = im[y][x]
            else:
                cop[y][x] = im[y][x+1]
    return cop

def argmin(a):
    amin = 0
    for i in range(len(a)):
        if(a[i] < a[amin]):
            amin = i
    return amin