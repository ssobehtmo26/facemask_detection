#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime

model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model=load_model('saved_model.h5')

img_width, img_height = 150,150

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade _frontalface_default.xml')

#cap=cv2.VideoCapture('video_mask.mp4')
cap=cv2.VideoCapture(0)
img_count_full=0

font=cv2.FONT_HERSHEY_SIMPLEX
org=(1,1)
class_label=''
fontScale=1
color=(255,0,0)
thickness=2

while True:
    img_count_full+=1
    response, color_img=cap.read()
    
    if response == False:
        break
        
    scale=50
    width=int(color_img.shape[1]*scale/100)
    height=int(color_img.shape[0]*scale/100)
    dim=(width,height)
    
    color_img=cv2.resize(color_img, dim, interpolation=cv2.INTER_AREA)
    
    gray_img=cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6)
    
    img_count=0
    for(x,y,w,h) in faces:
        org=(x-10,y-10)
        img_count+=1
        color_face=color_img[y:y+h,x:x+w]
        cv2.imwrite('input/%d%dface.jpg'%(img_count_full,img_count),color_face)
        img=load_img('input/%d%dface.jpg'%(img_count_full,img_count), target_size=(img_width,img_height))
        img=img_to_arr(img)
        img=np.expand_dim(img,axis=0)
        prediction=model.predict(img)
        if prediction==0:
            class_label='Mask'
            color=(0,0,255)
        else:
            class_label='No Mask'
            color=(0,0,255)
            
        cv2.rectangle(color_img, (x,y), (x+w,y+h), (255,0,0),3)
        cv2.putText(color_img, class_label, org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Face Mask Detection', color_img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

