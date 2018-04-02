from PIL import Image
import time
import numpy as np
import sys
import preprocessingCUDA as prep
import seamsCUDA as seams
from statistics import mean

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
    deb = time.time() * 1000
    im_array = np.array(cop)
    gradient = np.zeros((len(im_array),len(im_array[0])) , dtype=np.int16)
    prep.sobel(im_array,gradient)
    fin = time.time() * 1000
    sobel_time = fin - deb
    for i in range(int(sys.argv[2])):
        # print("{} ème iteration".format(i))

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


if __name__ == '__main__':
    main()
    