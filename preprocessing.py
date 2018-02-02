from PIL import Image
import math

def energy_map(im):
    conv1 = [[-1,0,1],[-2,0,2],[-1,0,1]]
    conv2 = [[-1,-2,-1],[0,0,0],[1,2,1]]
    cop = im.convert("L")
    pix = cop.load()
    map_ener = [[0]*im.size[0] for y in range(im.size[1])]
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            voisinnage = [[0,0,0],[0,0,0],[0,0,0]]
            i=0
            for yp in range(y-1,y+2):
                j=0
                for xp in range(x-1,x+2):
                    if 0<=xp<im.size[0] and 0<=yp<im.size[1]:
                        v = pix[xp,yp]
                        voisinnage[j][i]=v
                    j+=1
                i +=1

            Gx=convo3x3(conv1,voisinnage)
            Gy=convo3x3(conv2,voisinnage)
            G=math.sqrt(Gx**2+Gy**2)
            G = int(G) if int(G) <=255 else 255
            map_ener[y][x]=G
    return map_ener


def convo3x3(A,B):
    tot = 0
    for j in range(3):
        for i in range(3):
            tot += A[i][j]*B[i][j]
    return tot
