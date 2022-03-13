import os
from PIL import Image # Python Image Library
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
rec = cv2.face.LBPHFaceRecognizer_create()

base = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base, 'Images')

ide =0
lab_ids = {}

x_train = []
y_train = []

for root,dirs,files in os.walk(img_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			path = os.path.join(root,file)
			label = os.path.basename(root).replace(" ","-").lower()
			
			if not label in lab_ids:
				lab_ids[label] = ide
				ide += 1

			id_ = lab_ids[label]	

			img = Image.open(path).convert("L") #converts to gray scale
			img_array = np.array(img,"uint8")
			
			faces = face_cascade.detectMultiScale(img_array,scaleFactor = 1.5, minNeighbors = 5)

			for (x,y,w,h) in faces:
				region = img_array[y:y+h, x:x+w]
				x_train.append(region)
				y_train.append(id_)


with open('label.pickle', 'wb') as f:
	pickle.dump(lab_ids, f)

rec.train(x_train, np.array(y_train))
rec.save("trained.yml")
			

