from PIL import Image
def morpho_gradient(im):
    #im est en niveaux de gris
   
    pix= im.load()
    matrix=[[0]*im.size[1] for i in range(im.size[0])]
    for y in range(im.size[1]): 
        for x in range (im.size[0]):
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
                   
                    k = pix[i,j] #parcours les pixels 1 à 1
                   
                   
                    if k>max:
                        max=k
                    if k<min:
                        min=k
                   
            matrix[y][x]=max-min
    print(matrix)   
    return matrix         


def sc(im):
    im = Image.open("im.jpg").convert("L")    
    buf= im.load()
    cop=im.copy()
    buff=cop.load()  
                                                   
    m= morpho_gradient(im)
    m2= [[0]*len(m) for i in range(len(m[0]))]
   
   
   
    for x in range((len(m[0]))):
        m2[0][x]=m[0][x]
        for y in range(1,(len(m))-1):
           
                if (x==0):
                    m2[y][x]= m[y][x+1]
                
                elif (x==len(m[0])):
                    m2[y][x]=m[y][x-2]
                    
                else:
                    m2[y][x] =  m2[y][x]+ min(m2[y][x-1], m2[y][x], m2[y][x+1])
                
    print("7")
sc("im.jpg")