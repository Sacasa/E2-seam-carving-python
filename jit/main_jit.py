from PIL import Image
import time
import numpy as np
import sys
import preprocessing_jit as prep
import seams_jit as seams
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numba import autojit

def main():
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

    #using matrices  
    #https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array  

    im_array = np.array(img)

    deb = time.time() * 1000
    cop_array = np.array(cop)
    gradient = np.zeros((len(cop_array),len(cop_array[0])) , dtype=np.int16)
    prep.sobel_v2(cop_array,gradient)
    fin = time.time() * 1000
    sobel_time = fin - deb
    for i in range(int(sys.argv[2])):
        # print("{} ème iteration".format(i))

        deb = time.time() * 1000
        list_sims_coords = seams.dynamic_programming(gradient)
        fin = time.time() * 1000
        temps_seams.append(fin-deb)

        deb = time.time() * 1000
        im_array = seams.move_mat_rgb(im_array,list_sims_coords)
        fin = time.time() * 1000
        temps_dec_im.append(fin-deb)

        deb = time.time() * 1000
        gradient = seams.move_mat(gradient,list_sims_coords)
        fin = time.time() * 1000
        temps_dec_grad.append(fin-deb)


    deb = time.time() * 1000
    imag = create_im_fromarray(im_array)
    imag.save("images/result.png")
    fin = time.time() * 1000
    temps_sauv = fin-deb


    end = int(round(time.time() * 1000))
    print("Temps de calcul de sobel: {}".format(sobel_time))
    print("====================================================================")
    print("Temps moyen de calcul des seams: {}".format(mean(temps_seams)))
    print("Temps moyen de décalage de l'image : {}".format(mean(temps_dec_im)))
    print("Temps moyen de décalage du gradient : {}".format(mean(temps_dec_grad)))
    print("Temps moyen par étape : {}".format(mean(temps_seams) + mean(temps_dec_im) + mean(temps_dec_grad)))
    print("====================================================================")
    print("Temps de sauvegarde de l'image : {}".format(temps_sauv))
    print("Temps d'execution : {} ms".format(end-start))
    print("Taille originale : ({},{})".format(im.size[0],im.size[1]))
    print("Taille finale : ({},{})".format(img.size[0],img.size[1]))

    imag.show()

def create_im_fromarray(array):
    cop = Image.new("RGB",(array.shape[1],array.shape[0]))
    pix_cop = cop.load()
    for y in range(cop.size[1]):
        for x in range(cop.size[0]):
            liste = array[y][x]
            pix_cop[x,y] = (liste[0],liste[1],liste[2])
    return cop

if __name__ == '__main__':
    main()
    