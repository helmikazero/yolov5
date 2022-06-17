import cv2
import torch
import numpy as np


imgsz = (240,240)

stream = cv2.VideoCapture(0)
customyolov5s = torch.hub.load('','custom', path='weightHedect/nextgen_hedec_s2.pt', source='local')

#colorcode
clrYellow = (38,247,255)
clrGreen = (0,255,0)
clrRed = (0,0,255)

def score_frame(frame,model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    results = model(frame)
    mantap = results.pandas().xyxy[0]
    
    return mantap

def addCenterToPresultsDataFrame(p_presults):
    newcol_xycenter = []

    for i in range(len(p_presults)):
        newcol_xycenter.append((
            int((p_presults["xmin"][i]+p_presults["xmax"][i])/2),int((p_presults["ymin"][i]+p_presults["ymax"][i])/2)
        ))

    presults2 = p_presults
    presults2["xycenter"] = newcol_xycenter

    return presults2



def bb_intersection_over_union(presults, nameA,nameB):
	boxA = [presults['xmin'][nameA],presults['ymin'][nameA],presults['xmax'][nameA],presults['ymax'][nameA]]
	boxB = [presults['xmin'][nameB],presults['ymin'][nameB],presults['xmax'][nameB],presults['ymax'][nameB]]

	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def check_helmetWear(presults):
    head_list = []
    hardhat_list = []
    pair_result = []
    for i in range(len(presults)):
        if presults['name'][i] == "helmet":
            hardhat_list.append(i)
        elif presults['name'][i] == "head":
            head_list.append(i)

    print(head_list)
    print(hardhat_list)
    pair_result = []

    if len(hardhat_list) == 0:
        for i in range(len(head_list)):
            pair_result.append((head_list[i],-1,0))
        return pair_result

    for i in range(len(head_list)):
        # hardhat_iou_scores = []
        besthardhat = []
        for j in range(len(hardhat_list)):
            iou_result = bb_intersection_over_union(presults,head_list[i],hardhat_list[j])
            
            if len(besthardhat) == 0:
                besthardhat = [hardhat_list[j],iou_result]
                continue

            if besthardhat[1] < iou_result:
                besthardhat = [hardhat_list[j],iou_result]

        print(besthardhat)
        pair_result.append([head_list[i],besthardhat[0],besthardhat[1]])
    return pair_result


def print_result(m_presults,m_pairResult,plain_img,wearTresh):
    #colorCoding
    no_helmet_detected = 0
    printedimg = plain_img
    
    
    # print helmet
    for i in range(len(m_presults)):
        if m_presults["name"][i] == "helmet":
            printedimg = cv2.rectangle(printedimg, (int(m_presults['xmin'][i]),int(m_presults['ymin'][i])),(int(m_presults['xmax'][i]),int(m_presults['ymax'][i])),clrYellow,1)
            
            
    for i in range(len(m_pairResult)):
        if m_pairResult[i][2] < wearTresh:
            no_helmet_detected += 1
            printedimg = cv2.rectangle(printedimg, (int(m_presults['xmin'][m_pairResult[i][0]]),int(m_presults['ymin'][m_pairResult[i][0]])),(int(m_presults['xmax'][m_pairResult[i][0]]),int(m_presults['ymax'][m_pairResult[i][0]])),clrRed,2)
            printedimg = cv2.putText(printedimg,m_presults["name"][m_pairResult[i][0]],(int(m_presults['xmin'][m_pairResult[i][0]]),int(m_presults['ymin'][m_pairResult[i][0]])),cv2.FONT_HERSHEY_SIMPLEX,0.9,clrRed,2)
        else :
            printedimg = cv2.rectangle(printedimg, (int(m_presults['xmin'][m_pairResult[i][0]]),int(m_presults['ymin'][m_pairResult[i][0]])),(int(m_presults['xmax'][m_pairResult[i][0]]),int(m_presults['ymax'][m_pairResult[i][0]])),clrGreen,2)
            printedimg = cv2.putText(printedimg,m_presults["name"][m_pairResult[i][0]],(int(m_presults['xmin'][m_pairResult[i][0]]),int(m_presults['ymin'][m_pairResult[i][0]])),cv2.FONT_HERSHEY_SIMPLEX,0.9,clrGreen,2)

            printedimg = cv2.rectangle(printedimg, (int(m_presults['xmin'][m_pairResult[i][1]]),int(m_presults['ymin'][m_pairResult[i][1]])),(int(m_presults['xmax'][m_pairResult[i][1]]),int(m_presults['ymax'][m_pairResult[i][1]])),clrGreen,2)
            printedimg = cv2.putText(printedimg,m_presults["name"][m_pairResult[i][1]],(int(m_presults['xmin'][m_pairResult[i][1]]),int(m_presults['ymin'][m_pairResult[i][1]])),cv2.FONT_HERSHEY_SIMPLEX,0.9,clrGreen,2)
            
            printedimg = cv2.line(printedimg,m_presults['xycenter'][m_pairResult[i][0]],m_presults['xycenter'][m_pairResult[i][1]],clrGreen,2)

    return printedimg
    
            

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

        presults2 = addCenterToPresultsDataFrame(results)

        print("-----------------------THE RESULTS -------------------------------")
        print(presults2)
        print("-----------------------END OF RESULT---------------------------------")

        pair_result = check_helmetWear(results)
        img = print_result(presults2,pair_result,img,0.1)

        # img = printTheBoxYEAAAH(img,results)
        
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