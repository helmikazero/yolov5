import cv2
import torch
import numpy as np
import time


imgsz = (256,192)
# imgsz = (192,256)

stream = cv2.VideoCapture(0)

# stream.set(3,480)
# stream.set(4,384)



color = [(0,0,255),(0,255,0)]

# default_size = 640
# default_bbox_thicness = 2

# default_fps_pos = [320,30]
# default_fps_fsize = 1

# default_warning_pos = [0,640]
# default_warning_fsize = 1.8

# szmod = imgsz[0]/default_size

# def score_frame(frame,model):
    
#     results = model(frame)
#     mantap = results.pandas().xyxy[0]
    
#     return mantap
            

# def printHUD(fps,img, predresultpandas):
#     printed_img = img

#     no_helmet_count = 0
#     for i in range(len(predresultpandas)):

#         printed_img = cv2.rectangle(printed_img, (int(predresultpandas['xmin'][i]),int(predresultpandas['ymin'][i])),(int(predresultpandas['xmax'][i]),int(predresultpandas['ymax'][i])),color[int(predresultpandas['class'][i])],2)
#         printed_img = cv2.putText(printed_img,predresultpandas['name'][i],(int(predresultpandas['xmin'][i]),int(predresultpandas['ymin'][i])),cv2.FONT_HERSHEY_SIMPLEX,0.9,color[int(predresultpandas['class'][i])],2)
#         printed_img = cv2.putText(printed_img,str(predresultpandas['confidence'][i]),(int(predresultpandas['xmin'][i]),int(predresultpandas['ymax'][i])),cv2.FONT_HERSHEY_SIMPLEX,0.9,color[int(predresultpandas['class'][i])],2)

#         if predresultpandas['name'][i] == "no_helmet":
#             no_helmet_count += 1
        
#     # PRINT WARNING IF NO HELMET EXIST
#     if no_helmet_count > 0:
#         printed_img = cv2.putText(printed_img,"NO HELMET DETECTED",(int(default_warning_pos[0]*szmod),int(default_warning_pos[1]*szmod)),cv2.FONT_HERSHEY_SIMPLEX,default_warning_fsize*szmod,color[0],5)

#     # PRINT FPS
#     printed_img = cv2.putText(printed_img,fps,(int(320*szmod),int(30*szmod)),cv2.FONT_HERSHEY_SIMPLEX,0.9*szmod,color[0],2)

#     return printed_img




def main():
    old_time = 0
    new_time = 0

    model = torch.hub.load('','custom', path='weightHedect/hedec_yolov5s.pt', source='local')
    model.cuda()
    model.conf = 0.3
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # customyolov5s.to(device)

    while True:
        ret_val, img = stream.read()

        # img = crop_image_square(img)

        img = cv2.resize(img,imgsz)

        results = model(img, size=256)

        # print("-----------------------THE RESULTS -------------------------------")
        # print(results)
        # print("-----------------------END OF RESULT---------------------------------")

        new_time = time.time()
        fps = 1/(new_time-old_time)
        old_time = new_time

        fps = str(int(fps))

        img = cv2.putText(img,fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0, 255, 0),2)

        # img = printHUD(fps,img,results)
        
        cv2.imshow('Frame',img)

        if cv2.waitKey(1) == 27:
            break


# def crop_image_square(img):

#     resized_image = 0

#     if img.shape[0] < img.shape[1]:
#         resized_image = img[:int(img.shape[0]),int((img.shape[1]/2)-(img.shape[0]/2)):int((img.shape[1]/2)+(img.shape[0]/2))]

#     return resized_image

if __name__ == "__main__":
    main()