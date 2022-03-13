import cv2 # Library to work with images
import numpy as np  # Mathematics

net = cv2.dnn.readNet("/Users/xenox/Documents/Coaaadinggg/site/mysite/mysite/main/yolov3-tiny.weights","/Users/xenox/Documents/Coaaadinggg/site/mysite/mysite/main/yolov3-tiny.cfg") # Using a module in cv2 to read the weights and cfg file
classes = [] #Intitizalizing the variable classes to store all the classes

with open("/Users/xenox/Documents/Coaaadinggg/site/mysite/mysite/main/coco.names","r") as f: # Opening the file containing the classes
	classes = [line.strip() for line in f.readlines()] #Storing the values of the classes in the array classes

layers = net.getLayerNames() #Getting all the layesrs of the Neural Network
output = [layers[i[0] -1] for i in net.getUnconnectedOutLayers()] # Extracting the last layer containing the output

#Getting the image
img = cv2.imread("room_ser.jpg")# getting the image 
img = cv2.resize(img,None,fx = 0.4,fy = 0.4)#Resizing the imeage
h,w,c = img.shape # Getting the height and width and color channels of the image

#cCreating the blob
blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop = False)
for b in blob:
	for i,bb in enumerate(b):
		print("sdfghjk")

net.setInput(blob)
out = net.forward(output)

class_ids = []
boxex = []
confidences = []
for o in out: 
	for e in o:
		points = e[5:]
		class_id = np.argmax(points)
		confidence = points[class_id]
		if confidence > 0.5:
			center_x = int(e[0] * w)
			center_y = int(e[1] * h)
			wid = int(e[2] * w)
			hei = int(e[3] * h)

			#Drawing the rectangles
			x = int(center_x - wid / 2)
			y = int(center_y - hei / 2)
			

			boxex.append([x,y,wid,hei])
			confidences.append(float(confidence))
			class_ids.append(class_id)

total_objects = len(boxex)
font = cv2.FONT_HERSHEY_PLAIN
print(f"number of detections {total_objects}")
for i in range(total_objects):
	
	x,y,wid,hei = boxex[i]
	label = str(classes[class_ids[i]])
	print(label)
	cv2.rectangle(img,(x,y),(x+wid,y+hei),(255,0,0),1)
	cv2.putText(img,label,(x,y + 20),font,1,(0,255,0),1)


cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

