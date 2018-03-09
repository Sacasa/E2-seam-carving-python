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
            a=x-1
            b= x+1
            c= y-1
            d= y+1
            if (a<0):
                a=0
            if (b>=im.size[0]):
                b=im.size[0]-1
            if (c<0):
                c=0
            if (d>=im.size[1]):
                d=im.size[1]-1
            max=0
            min=255
            for i in range (a,b+1):
                for j in range (c,d+1):
                   
                    k = pix[i,j] #parcourt les pixels 1 à 1
                   
                   
                    if k>max:
                        max=k
                    if k<min:
                        min=k
              
            G=int(max-min)
            
            
               
    
           
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