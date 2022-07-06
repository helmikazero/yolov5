import argparse
import cv2
import numpy as np
import time
import os
import keyboard


def main(source,saveat,buffer):
    old_time = 0
    new_time = 0

    filename = os.path.basename(source)[:-4]

    stream = cv2.VideoCapture(source)

    progress = 0
    lenght = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    while(stream.isOpened()):
        ret_val, img = stream.read()
        # img = crop_image_square(img)
        # img = cv2.resize(img,imgsz)
        if ret_val == True: 
            
            cv2.imshow(filename,img)

            if progress % buffer == 0:
                saveframe(img,filename,saveat)

            if keyboard.is_pressed('space'):
                saveframe(img,filename,saveat)

            if cv2.waitKey(1) == 27:
                break

            progress = progress+1
            print(str(round((progress/lenght)*100,2))+"%"+"  "+str(progress)+"/"+str(lenght))
            # if cv2.waitKey(1) == 27:
            #     break
        else:
            break

    # stream.released()

    cv2.destroyAllWindows()


def saveframe(img,filename,saveat):
    savepath = uniquify(saveat+filename+'.png')
    savestatus = cv2.imwrite(savepath,img)
    print(savepath+str(savestatus))


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def input_bool(inpt):
    if inpt == "0":
        return False
    elif inpt == "1":
        return True

def crop_image_square(img):

    resized_image = 0

    if img.shape[0] < img.shape[1]:
        resized_image = img[:int(img.shape[0]),int((img.shape[1]/2)-(img.shape[0]/2)):int((img.shape[1]/2)+(img.shape[0]/2))]

    return resized_image

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='address')
    parser.add_argument('--saveat', type=str, help='wheretosave')
    parser.add_argument('--buffer', type=int, help='how many image before save')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    main(**vars(parse_opt()))