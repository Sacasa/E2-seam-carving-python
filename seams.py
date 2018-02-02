from PIL import Image


def dynamic_programming(mat):
    cop = list(mat)
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


    start = cop[0].index(min(cop[0]))
    seam = [(start,0)]
    x = start
    dx = [-1,0,1]
    for y in range(len(cop)-1):
        center = cop[y+1][x]
        if x == 0:
            values = [250000,center,cop[y+1][x+1]]
        elif x == len(cop[0])-1:
            values = [cop[y+1][x-1],center,250000]
        else:
            values = [cop[y+1][x-1],center,cop[y+1][x+1]]
        x += dx[values.index(min(values))]
        seam.append((x,y+1))

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