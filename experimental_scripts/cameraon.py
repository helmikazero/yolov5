import cv2
import torch
import numpy as np
import time

imgsz = (640,640)
stream = cv2.VideoCapture(0)




def main():
    old_time = 0
    new_time = 0

    while True:
        ret_val, img = stream.read()

        # img = crop_image_square(img)

        # img = cv2.resize(img,imgsz)

        # print("-----------------------THE RESULTS -------------------------------")
        # print(results)
        # print("-----------------------END OF RESULT---------------------------------")

        new_time = time.time()
        fps = 1/(new_time-old_time)
        old_time = new_time

        fps = str(int(fps))

        reso_tx = "x :"+str(img.shape[0])+" y:"+str(img.shape[1])

        img = cv2.putText(img,reso_tx,(0,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 0),2)

        img = cv2.putText(img,fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 0),2)
        
        cv2.imshow('Frame',img)

        if cv2.waitKey(1) == 27:
            break


def crop_image_square(img):

    resized_image = 0

    if img.shape[0] < img.shape[1]:
        resized_image = img[:int(img.shape[0]),int((img.shape[1]/2)-(img.shape[0]/2)):int((img.shape[1]/2)+(img.shape[0]/2))]

    return resized_image

if __name__ == "__main__":
    main()