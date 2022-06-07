import cv2
import torch
import numpy as np


imgsz = (240,240)

stream = cv2.VideoCapture(0)
customyolov5s = torch.hub.load('','custom', path='weightHedect/nexgen_hedec_s1.pt', source='local')

def score_frame(frame,model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = model(frame)
    mantap = results.pandas().xyxy[0]
    
    return mantap

def printTheBoxYEAAAH(img,predresultinpandas):
    printed_img = img
    for i in range(len(predresultinpandas)):
        printed_img = cv2.rectangle(printed_img, (int(predresultinpandas['xmin'][i]),int(predresultinpandas['ymin'][i])),(int(predresultinpandas['xmax'][i]),int(predresultinpandas['ymax'][i])),(0, 255, 0),2)
        printed_img = cv2.putText(printed_img,predresultinpandas['name'][i],(int(predresultinpandas['xmin'][i]),int(predresultinpandas['ymin'][i])),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 0),2)
        printed_img = cv2.putText(printed_img,str(predresultinpandas['confidence'][i]),(int(predresultinpandas['xmax'][i]),int(predresultinpandas['ymax'][i])),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 0),2)
    return printed_img

def main():
    while True:
        ret_val, img = stream.read()

        img = crop_image_square(img)

        results = score_frame(img,customyolov5s)
        img = printTheBoxYEAAAH(img,results)

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