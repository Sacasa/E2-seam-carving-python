from PIL import Image
import math
import numpy as np
from numba import cuda
from numba import autojit
from numba import *

@autojit
def convo_n(A,B,n):
    tot = 0
    for j in range(n):
        for i in range(n):
            tot += A[i][j]*B[i][j]
    return tot

convo_gpu = cuda.jit(restype=f8, argtypes=[uint16[:,:], uint16[:,:],uint8], device=True)(convo_n)


@autojit
def sobel(im_array,map_ener):
    conv1 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype =np.int16)
    conv2 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype =np.int16)

    for x in range(len(im_array[0])):
        for y in range(len(im_array)):
            voisinnage = np.zeros((3,3), dtype= np.int16)
            i=0
            for yp in range(y-1,y+2):
                j=0
                for xp in range(x-1,x+2):
                    if 0<=xp<len(im_array[0]) and 0<=yp<len(im_array):
                        v = im_array[yp][xp]
                        voisinnage[j][i]=v
                    j+=1
                i +=1

            Gx=convo_n(conv1,voisinnage,3)
            Gy=convo_n(conv2,voisinnage,3)
            G=math.sqrt(Gx**2+Gy**2)
            G = int(G) if int(G) <=255 else 255
            map_ener[y][x]=int(G)

@cuda.jit(argtypes=[uint16[:,:], uint16[:,:],uint16[:,:], uint16[:,:], uint16[:,:]])
def sobel_kernel(im_array,map_ener,conv1,conv2,voisinnage):

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX,len(im_array[0]),gridX):
        for y in range(startY,len(im_array),gridY):
            i=0
            for yp in range(y-1,y+2):
                j=0
                for xp in range(x-1,x+2):
                    if 0<=xp<len(im_array[0]) and 0<=yp<len(im_array):
                        v = im_array[yp][xp]
                        voisinnage[j][i]=v
                    j+=1
                i +=1

            Gx=convo_gpu(conv1,voisinnage,3)
            Gy=convo_gpu(conv2,voisinnage,3)
            G=math.sqrt((np.float64(Gx))**2+(np.float64(Gy))**2)
            G = int(G) if int(G) <=255 else 255
            map_ener[y][x]=int(G)