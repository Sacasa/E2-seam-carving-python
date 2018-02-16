def morpho_gradient(im):
    #im est en niveaux de gris
    im=image.jpg
    pix= im.load()
    cop = im.copy()
    pix2= cop.load()
    G = [[0]*im.size[0] for y in range(im.size[1])]
    for y in range(im.size[1]):  
        for x in range (im.size[0]):
          
            max=0
            min=255
            for i in range (x-1,x+2):
                for j in range (y-1,y+2):
                    
                    k = pix[i,j] #parcours les pixels 1 à 1
                    
                    
                    if k>max:
                        max=k
                    if k<min:
                        min=k
                    
    G[y][x]=max-min           
            
           
    return G

def seam_carving(im):
    M = morpho_gradient(im)
    y = len(M[0])
    M2 = [[0]*im.size[0] for y in range(im.size[1])]
    for i in range(1,y):
        M2[0][i] = M[0]
        for j in range(1,y):
            M2[j][i] =  M2[j][i]+ min(M2[i-1, j], M2[i, j], M2[i+1, j])
            M2.remove(min(M2[i-1, j], M2[i, j], M2[i+1, j]))
    print("7")
