import cv2 # Library to work with images
import numpy as np  # Mathematics
import time
net = cv2.dnn.readNet("/Users/xenox/Documents/Coaaadinggg/site/mysite/mysite/main/yolov3-tiny.weights","/Users/xenox/Documents/Coaaadinggg/site/mysite/mysite/main/yolov3-tiny.cfg") # Using a module in cv2 to read the weights and cfg file
classes = [] #Intitizalizing the variable classes to store all the classes



with open("/Users/xenox/Documents/Coaaadinggg/site/mysite/mysite/main/coco.names","r") as f: # Opening the file containing the classes
	classes = [line.strip() for line in f.readlines()] #Storing the values of the classes in the array classes

layers = net.getLayerNames() #Getting all the layesrs of the Neural Network
output = [layers[i[0] -1] for i in net.getUnconnectedOutLayers()] # Extracting the last layer containing the output

#Getting the image
capture = cv2.VideoCapture(0)
# Getting the height and width and color channels of the image
now = time.time()
frameID = 0
font = cv2.FONT_HERSHEY_PLAIN
#cCreating the blob
while True:
    _, frame = capture.read()
    frameID += 1
    h,w,c = frame.shape 
    blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop = False)
   
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
    
    print(f"number of detections {total_objects}")
    for i in range(total_objects):
    	
    	x,y,wid,hei = boxex[i]
    	label = str(classes[class_ids[i]])
    	print(label)
    	cv2.rectangle(frame,(x,y),(x+wid,y+hei),(255,0,0),1)
    	cv2.putText(frame,label,(x,y + 20),font,1,(0,255,0),1)
    
    
    elapTime = time.time() - now
    fps = frameID / elapTime
    cv2.putText(frame,"FPS : " + str(fps), (10,30), font, 2,(0,255,0), 2)
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()		

        
