from logging import warning
import cv2
import torch
import numpy as np
import time
import os
import argparse


models_name = [
    'hedec_pretrain_N',
    'hedec_pretrain_S',
    'hedec_pretrain_M',
    'hedec_pretrain_L',
    'hedec_pure_N',
    'hedec_pure_S',
    'hedec_pure_M',
    'hedec_pure_L'
]

color = [(0,0,255),(0,255,0)]

default_bbox_thicness = 2

default_fps_pos = (320,0)
default_fps_fsize = 1

default_warning_pos = (0,600)
default_warning_fsize = 1.8

fps_textsz = cv2.getTextSize('00',cv2.FONT_HERSHEY_SIMPLEX,0.9,2)[0]

warning_text = 'NO_HELMET DETECTED'
textsize = cv2.getTextSize(warning_text,cv2.FONT_HERSHEY_SIMPLEX,default_warning_fsize,5)[0]
print(textsize[1])

savevid = True


def score_frame(frame,model):
    results = model(frame, size=640)
    mantap = results.pandas().xyxy[0]
    
    return mantap
            

def printHUD(fps,img, predresultpandas):
    printed_img = img

    no_helmet_count = 0
    for i in range(len(predresultpandas)):

        printed_img = cv2.rectangle(printed_img, (int(predresultpandas['xmin'][i]),int(predresultpandas['ymin'][i])),(int(predresultpandas['xmax'][i]),int(predresultpandas['ymax'][i])),color[int(predresultpandas['class'][i])],2)
        printed_img = cv2.putText(printed_img,predresultpandas['name'][i],(int(predresultpandas['xmin'][i]),int(predresultpandas['ymin'][i])),cv2.FONT_HERSHEY_SIMPLEX,0.9,color[int(predresultpandas['class'][i])],2)
        printed_img = cv2.putText(printed_img,str(round(predresultpandas['confidence'][i],2)),(int(predresultpandas['xmin'][i]),int(predresultpandas['ymax'][i])),cv2.FONT_HERSHEY_SIMPLEX,0.9,color[int(predresultpandas['class'][i])],2)

        if predresultpandas['name'][i] == "no_helmet":
            no_helmet_count += 1
        
    # PRINT WARNING IF NO HELMET EXIST
    if no_helmet_count > 0:
        printed_img = cv2.putText(printed_img,"NO_HELMET DETECTED",(default_warning_pos[0],printed_img.shape[0]-textsize[1]),cv2.FONT_HERSHEY_SIMPLEX,default_warning_fsize,color[0],5)
        TRIGGER_ALARM()

    # PRINT FPS
    printed_img = cv2.putText(printed_img,fps,(int(printed_img.shape[1]/2),fps_textsz[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,color[0],2)

    return printed_img

def TRIGGER_ALARM():
    print("ALARM FOR NO HELMET HAS BEEN TRIGGERED")


def main(weights,source,saveas):
    old_time = 0
    new_time = 0

    # print(device)

    model = torch.hub.load('','custom', path=weights, source='local')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    stream = cv2.VideoCapture(source)

    default_path = saveas
    check_path = uniquify(default_path)

    result = cv2.VideoWriter(filename=check_path, 
                         fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                         fps=stream.get(cv2.CAP_PROP_FPS), frameSize=(int(stream.get(3)),int(stream.get(4))))
    progress = 0
    lenght = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    while(stream.isOpened()):
        ret_val, img = stream.read()
        # img = crop_image_square(img)
        # img = cv2.resize(img,imgsz)
        if ret_val == True: 
            results = score_frame(img,model)

            new_time = time.time()
            fps = 1/(new_time-old_time)
            old_time = new_time

            fps = str(int(fps))

            img = printHUD(fps,img,results)

            if savevid:
                result.write(img)
            
            progress = progress+1
            print(str(round((progress/lenght)*100,2))+"%"+"  "+str(progress)+"/"+str(lenght))
            # if cv2.waitKey(1) == 27:
            #     break
        else:
            break


    if savevid:
        stream.release()
        result.release()

    cv2.destroyAllWindows()

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def crop_image_square(img):

    resized_image = 0

    if img.shape[0] < img.shape[1]:
        resized_image = img[:int(img.shape[0]),int((img.shape[1]/2)-(img.shape[0]/2)):int((img.shape[1]/2)+(img.shape[0]/2))]

    return resized_image


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='model path')
    parser.add_argument('--source', type=str, help='address')
    parser.add_argument('--saveas',type=str,help='where to save')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    main(**vars(parse_opt()))