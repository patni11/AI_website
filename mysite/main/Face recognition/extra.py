     ''' 
      labels = {"name":1}
      with open(f'{path}label.pickle','rb') as f:
        alabels = pickle.load(f)
        labels = {a:b for b,a in alabels.items()}
      # Capture frame-by-frame
      '''
      

     ''' 
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      faces = face.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 5)

      for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+h]
        roi_c = frame[y:y+h, x:x+h]

        idd,conf = rec.predict(roi)
        if conf >= 75:
          print(idd)
          print(labels[idd])

        cv2.putText(frame,labels[idd],(x,y + 20),font,1,(0,255,0),2)  
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      # Display the resulting frame
     '''  