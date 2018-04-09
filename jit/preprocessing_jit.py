from PIL import Image
import math
import numpy as np
from numba import autojit

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
                    
                    v = im_array[yp][xp]
                    voisinnage[j][i]=v
                    j+=1
                i +=1

            Gx=convo_n(conv1,voisinnage,3)
            Gy=convo_n(conv2,voisinnage,3)
            G=math.sqrt(Gx**2+Gy**2)
            G = int(G) if int(G) <=255 else 255
            map_ener[y][x]=int(G)

@autojit
def sobel_points(im_array,points,gradient):
    convx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype =np.int16)
    convy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype =np.int16)

    #calculer sobel en x-1 et x uniquement
    for point in points:
        Gx=0
        Gy=0
        for xp in range(point[0]-1,point[0]+1):
            if 0<=xp<x_max and 0<=yp<y_max: 
                dx = xp - x + 1
                dy = yp -y + 1
                Gx += convx[dy][dx]*im_array[yp][xp]
                Gy += convy[dy][dx]*im_array[yp][xp]
        
        G=math.sqrt(Gx**2+Gy**2)
        G = int(G) if int(G) <=255 else 255
        map_ener[point[1]][point[0]]=int(G)



@autojit
def sobel_v2(im_array,map_ener):
    convx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype =np.int16)
    convy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype =np.int16)
    Gx = 0
    Gy = 0
    x_max = len(im_array[0])
    y_max = len(im_array)
    for x in range(x_max):
        for y in range(y_max):
            for yp in range(y-1,y+2):
                for xp in range(x-1,x+2):
                    if 0<=xp<x_max and 0<=yp<y_max: 
                        dx = xp - x + 1
                        dy = yp -y + 1
                        Gx += convx[dy][dx]*im_array[yp][xp]
                        Gy += convy[dy][dx]*im_array[yp][xp]

            G=math.sqrt(Gx**2+Gy**2)
            G = int(G) if int(G) <=255 else 255
            map_ener[y][x]=int(G)


@autojit
def convo_n(A,B,n):
    tot = 0
    for j in range(n):
        for i in range(n):
            tot += A[i][j]*B[i][j]
    return tot
