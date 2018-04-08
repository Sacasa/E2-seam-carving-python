from PIL import Image
import math

def sobel(im):
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

            Gx=convo_n(conv1,voisinnage,3)
            Gy=convo_n(conv2,voisinnage,3)
            G=math.sqrt(Gx**2+Gy**2)
            G = int(G) if int(G) <=255 else 255
            map_ener[y][x]=int(G)
    return map_ener

def sobel_v2(im):
    convx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    convy = [[-1,-2,-1],[0,0,0],[1,2,1]]
    cop = im.convert("L")
    pix = cop.load()
    Gx=0
    Gy=0
    map_ener = [[0]*im.size[0] for y in range(im.size[1])]
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            for yp in range(y-1,y+2):
                for xp in range(x-1,x+2):
                    if 0<=xp<im.size[0] and 0<=yp<im.size[1]:
                        dx = xp - x + 1
                        dy = yp -y + 1
                        Gx += convx[dy][dx]*pix[xp,yp]
                        Gy += convy[dy][dx]*pix[xp,yp]

            G=math.sqrt(Gx**2+Gy**2)
            G = int(G) if int(G) <=255 else 255
            map_ener[y][x]=int(G)
    return map_ener

def sobel_img(im):
    conv1 = [[-1,0,1],[-2,0,2],[-1,0,1]]
    conv2 = [[-1,-2,-1],[0,0,0],[1,2,1]]
    im=im.convert("L")
    pix = im.load()
    cop = im.copy()
    pix_cop = cop.load()
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

            Gx=convo_n(conv1,voisinnage,3)
            Gy=convo_n(conv2,voisinnage,3)
            G=math.sqrt(Gx**2+Gy**2)
            G= int(G) if int(G) <=255 else 255
            pix_cop[x,y]=G
    return cop

def convo_n(A,B,n):
    tot = 0
    for j in range(n):
        for i in range(n):
            tot += A[i][j]*B[i][j]
    return tot
