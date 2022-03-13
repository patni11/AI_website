import numpy as np
import cv2
import pickle

face = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("trained.yml")
font = cv2.FONT_HERSHEY_PLAIN
labels = {"name":1}
with open('label.pickle','rb') as f:
	alabels = pickle.load(f)
	labels = {a:b for b,a in alabels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 5)

    for (x,y,w,h) in faces:
    	roi = gray[y:y+h, x:x+h]
    	roi_c = frame[y:y+h, x:x+h]

    	idd,conf = rec.predict(roi)
    	if conf >= 75:
    		
    		print(labels[idd])

    	print(conf)	
    	cv2.putText(frame,labels[idd],(x,y + 20),font,1,(0,255,0),2)	
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       	break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()