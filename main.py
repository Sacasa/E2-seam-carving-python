from PIL import Image
import time
import numpy as np
import sys
import preprocessing
import seams
import imageio

if __name__ == '__main__':
    start = int(round(time.time() * 1000))
    name = sys.argv[1]
    im = Image.open(name)
    cop = im.convert("L")
    pix = im.load()
    img = im.copy()
    images_gif=[]

    for i in range(200):
        print("{} ème iteration".format(i))
        gradient = preprocessing.energy_map(img)
        list_sims_coords = seams.dynamic_programming(gradient)
        img = seams.move(img,list_sims_coords)
        cop2 = img.copy()
        pix_cop = cop2.load()
        for coord in list_sims_coords:
            pix_cop[coord] = (255,0,0)
        cop2.save("images/img{}.jpg".format(i))
        images_gif.append(imageio.imread("images/img{}.jpg".format(i)))

    end = int(round(time.time() * 1000))
    imageio.mimsave('images/movie.gif', images_gif)
    print("")
    print("Temps d'execution : {} ms".format(end-start))
    print("Temps moyen par iterartion : {} ms".format((end-start)/200))
    print("Taille originale : ({},{})".format(im.size[0],im.size[1]))
    print("Taille originale : ({},{})".format(img.size[0],img.size[1]))

    img.show()
