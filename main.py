from PIL import Image
import time
import numpy as np
import sys
import preprocessing
import seams
from statistics import mean

if __name__ == '__main__':
    start = int(round(time.time() * 1000))
    name = sys.argv[1]
    im = Image.open(name)
    cop = im.convert("L")
    pix = im.load()
    img = im.copy()
    images_gif=[]

    temps_seams=[]
    temps_dec_im = []
    temps_dec_grad= []
    temps_sauv = []

    # gradient = preprocessing.sobel(cop)
    # for i in range(int(sys.argv[2])):
    #     print("{} ème iteration".format(i))
    #     list_sims_coords = seams.dynamic_programming(gradient)
    #     img = seams.move(img,list_sims_coords)
    #     gradient = seams.move_mat(gradient,list_sims_coords)
    #     img.save("images/img{}.png".format(i))

    deb = time.time() * 1000
    gradient = preprocessing.sobel(cop)
    fin = time.time() * 1000
    sobel_time = fin - deb
    for i in range(int(sys.argv[2])):

        deb = time.time() * 1000
        list_sims_coords = seams.dynamic_programming(gradient)
        fin = time.time() * 1000
        temps_seams.append(fin-deb)

        deb = time.time() * 1000
        img = seams.move(img,list_sims_coords)
        fin = time.time() * 1000
        temps_dec_im.append(fin-deb)

        deb = time.time() * 1000
        gradient = seams.move_mat(gradient,list_sims_coords)
        fin = time.time() * 1000
        temps_dec_grad.append(fin-deb)


        deb = time.time() * 1000
        img.save("images/img{}.png".format(i))
        fin = time.time() * 1000
        temps_sauv.append(fin-deb)

    # sobel = preprocessing.sobel_img(im).convert("L")

    # for i in range(int(sys.argv[2])):
    #     print("{} ème iteration".format(i))

    #     sobel_pix = sobel.load()
    #     gradient =[[0]*img.size[0] for y in range(img.size[1])]
    #     for x in range(img.size[0]):
    #         for y in range(im.size[1]):
    #             gradient[y][x] = sobel_pix[x,y]
    #     list_sims_coords = seams.dynamic_programming(gradient)
    #     img = seams.move(img,list_sims_coords)
    #     sobel = seams.move_l(sobel,list_sims_coords)
    #     img.save("images2/img{}.png".format(i))

    end = int(round(time.time() * 1000))
    print("====================================================================")
    print("Temps moyen de calcul de sobel: {}".format(sobel_time))
    print("Temps moyen de calcul des seams: {}".format(mean(temps_seams)))
    print("Temps moyen de décalage de l'image : {}".format(mean(temps_dec_im)))
    print("Temps moyen de décalage du gradient : {}".format(mean(temps_dec_grad)))
    print("Temps moyen de sauvegarde de l'image (ne restera pas) : {}".format(mean(temps_sauv)))
    print("====================================================================")
    print("Temps d'execution : {} ms".format(end-start))
    print("Temps moyen par iterartion : {} ms".format((end-start)/int(sys.argv[2])))
    print("Taille originale : ({},{})".format(im.size[0],im.size[1]))
    print("Taille finale : ({},{})".format(img.size[0],img.size[1]))

    img.show()
