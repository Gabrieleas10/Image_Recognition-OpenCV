import numpy as np
import cv2

face_file = 'E:\Projects\Image_Recognition\haarcascades\haarcascade_frontalface_default.xml'
eye_file = 'E:\Projects\Image_Recognition\haarcascades\haarcascade_eye.xml'
smile_file = 'E:\Projects\Image_Recognition\haarcascades\haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(face_file)
eye_cascade = cv2.CascadeClassifier(eye_file) 
smile_cascade = cv2.CascadeClassifier(smile_file) 

image = cv2.imread('E:\Projects\Image_Recognition\images\smilepic.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces  = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(image , (x,y),(x + w , y + h), (255,0,0),2)
    smiles  = smile_cascade.detectMultiScale(gray , 1.8, 25)
    for (x1,y1,w1,h1) in smiles:
        cv2.rectangle(image , (x1,y1),(x1 + w1 , y1 + h1), (0,0,255),2)

cv2.imshow('Smile Detect' ,  image)
cv2.waitKey(0)

    

