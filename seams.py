from PIL import Image
import numpy as np


def dynamic_programming(mat):
    cop = [[0]*len(mat[0]) for y in range(len(mat))]
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

    start = cop[-1].index(min(cop[-1]))
    seam = [(start,len(mat)-1)]
    x = start
    dx = [-1,0,1]
    for y in range(len(cop)-1,0,-1):
        center = cop[y-1][x]
        if x == 0:
            values = [250000,center,cop[y-1][x+1]]
        elif x == len(cop[0])-1:
            values = [cop[y-1][x-1],center,250000]
        else:
            values = [cop[y-1][x-1],center,cop[y-1][x+1]]
        x += dx[values.index(min(values))]
        seam.append((x,y-1))

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

def move_mat(im,list):
    cop = np.zeros((len(im),len(im[0])-1), dtype=np.int16)
    for y in range(len(im)):
        for x in range(len(cop[0])):
            if x < list[y][0]:
                cop[y][x] = im[y][x]
            else:
                cop[y][x] = im[y][x+1]
    return cop

