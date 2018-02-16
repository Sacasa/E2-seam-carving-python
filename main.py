from PIL import Image
import time
import numpy as np
import sys
import preprocessing
import seams

if __name__ == '__main__':
    start = int(round(time.time() * 1000))
    name = sys.argv[1]
    im = Image.open(name)
    cop = im.convert("L")
    pix = im.load()
    img = im.copy()
    images_gif=[]

    for i in range(int(sys.argv[2])):
        print("{} ème iteration".format(i))
        gradient = preprocessing.sobel(img)
        list_sims_coords = seams.dynamic_programming(gradient)
        img = seams.move(img,list_sims_coords)
        cop2 = img.copy()
        pix_cop = cop2.load()
        for coord in list_sims_coords:
            pix_cop[coord] = (255,0,0)
        cop2.save("images/img{}.png".format(i))

    # sobel = preprocessing.sobel_img(im).convert("L")
    #
    # for i in range(int(sys.argv[2])):
    #     print("{} ème iteration".format(i))
    #
    #     sobel_pix = sobel.load()
    #     gradient =[[0]*img.size[0] for y in range(img.size[1])]
    #     for x in range(img.size[0]):
    #         for y in range(im.size[1]):
    #             gradient[y][x] = sobel_pix[x,y]
    #     list_sims_coords = seams.dynamic_programming(gradient)
    #     img = seams.move(img,list_sims_coords)
    #     sobel = seams.move_l(sobel,list_sims_coords)
    #     cop2 = img.copy()
    #     pix_cop = cop2.load()
    #     for coord in list_sims_coords:
    #         pix_cop[coord] = (255,0,0)
    #     cop2.save("images2/img{}.png".format(i))
    #     sobel.save("images2/sobel{}.png".format(i))

    end = int(round(time.time() * 1000))
    print("")
    print("Temps d'execution : {} ms".format(end-start))
    print("Temps moyen par iterartion : {} ms".format((end-start)/int(sys.argv[2])))
    print("Taille originale : ({},{})".format(im.size[0],im.size[1]))
    print("Taille finale : ({},{})".format(img.size[0],img.size[1]))

    img.show()
