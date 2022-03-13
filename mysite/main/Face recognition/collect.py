import numpy as np 
import cv2
import os
import time

vid = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

label = input("Who are you?       ").lower()
base = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base, 'Images')

if os.path.isdir(f'Images/{label}'):
	print("Already Exists.")
else:
	os.mkdir(f'Images/{label}')	
	a = 1
	
	while(True):
		if a <= 10:
			ret,frame = vid.read()
			cv2.imshow("frame",frame)

			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			faces = face.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 5)

			for (x,y,w,h) in (faces):
				
				if (x,y,w,h) != (0,0,0,0) and a <= 10:
					cv2.imwrite(f'Images/{label}/{a}.jpg',frame)
					print(a)
					a += 1
					time.sleep(1)

				else:
					break	

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			print("completed")
			img = cv2.imread("/Users/xenox/Documents/Coaaadinggg/site/mysite/mysite/main/Face recognition/Images/COMPLETED.png")
			cv2.imshow("image",img)
			time.sleep(1)
			break


vid.release()
cv2.destroyAllWindows()		