import numpy as np 
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict as ddict

model = load_model("./Models/localization_model.hdf5")


def loadImage(path):
    img = load_img(path,color_mode="grayscale")
    img = img_to_array(img)
    return img
    #plt.imshow(img)
    #plt.show()

########################################################################################################################

def getPredictions(img):
    global model
    pred = model.predict(img[np.newaxis,:,:,:])
    pred = pred.squeeze()
    ind = pred.argmax(axis=2)
    conf = pred.max(axis=2)
    return ind, conf

########################################################################################################################
#
#def group(boxes):
#    pl = 0
#    nl = 1
#    boxes.sort(key = lambda box:(box[0],box[1]))
#    while pl!=nl:
#        for i in range(1,len(boxes))
#
#
########################################################################################################################

def expansion(bounding_boxes, slide_x = 1, slide_y=1):
    nb_boxes = len(bounding_boxes)   
    idx = 0
    while idx < nb_boxes-1:
        y1, x1, box_size_y1, box_size_x1 = bounding_boxes[idx]
        y2, x2, box_size_y2, box_size_x2 = bounding_boxes[idx+1]      
        if (((x1 + box_size_x1 + slide_x >= x2) or (x2 + box_size_x2 + slide_x >= x1))
            and (y1 + box_size_y1 + slide_y >= y2)):
            x = min(x1, x2)
            y = min(y1, y2)
            box_size_y = max(y2+box_size_y2, y1+box_size_y1) - min(y1,y2)
            box_size_x = max(x2+box_size_x2, x1+box_size_x1) - min(x1,x2)
            bounding_boxes[idx]= np.array([y, x, box_size_y, box_size_x])
            bounding_boxes = np.delete(bounding_boxes, idx+1, axis = 0 )
            nb_boxes = len(bounding_boxes)   
        else:
            idx+=1

    return bounding_boxes

########################################################################################################################

def getBoxes(ind,conf):
    x,y = ind.shape
    cb = []
    for i in range(x):
        for j in range(y):
            if ind[i][j]==1 and conf[i][j]==1:
                cb.append([i*4,j*4,28,28])
                #print(1 if ind[i][j]==1 else' ',end='')
        #print()

    bound_box = expansion(cb)
    return bound_box

########################################################################################################################

def drawBoxes(img,boxes):
    test = img[:,:,0].astype(np.float32)

    test = cv2.cvtColor(test.copy(), cv2.COLOR_GRAY2BGR)
    #for x,y,sx,sy in cb:
    #        test = cv2.rectangle(test,(y,x),(y+sy,x+sx),(255,0,0),1)
    for x,y,sx,sy in boxes:
            test = cv2.rectangle(test,(y,x),(y+sy,x+sx),(0,255,0),1)

    fig=plt.figure(figsize=(12,8), dpi= 200, facecolor='w', edgecolor='k')
    plt.imshow(test/255)
    plt.show()


########################################################################################################################

def getSentenceImg(img,boxes):
    sentences = []
    for x1,y1,w,h in boxes:
        x = img[x1:w+x1,y1:y1+h,0]/255
        x[x<0.25]=0
        while True:
            if np.max(x[0,:])>0:
                break
            x = x[1:,:]
        while True:
            if np.max(x[-1,:])>0:
                break
            x = x[:-1,:]

        if x.shape[0]<28:
            w = (28-x.shape[0])//2
            x = np.pad(x,[(w,w)])
        sentences.append(x)
    return sentences

########################################################################################################################
def getSentence(path,show=False):
    img = loadImage(path)
    
    if show:
        fig=plt.figure(figsize=(12,8), dpi= 200, facecolor='w', edgecolor='k')
        plt.imshow(img)
        plt.show()

    ind, conf = getPredictions(img)
    boxes = getBoxes(ind,conf)

    if show:
        drawBoxes(img,boxes)

    sentences = getSentenceImg(img,boxes)
    if show:
        for sentence in sentences:
            fig=plt.figure(figsize=(12,8), dpi= 200, facecolor='w', edgecolor='k')
            plt.imshow(sentence)
            plt.show()
    return sentences




getSentence("./TestData/9.jpg",True)
