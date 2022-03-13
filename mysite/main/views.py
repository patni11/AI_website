from django.shortcuts import render, redirect
from django.http import HttpResponse,StreamingHttpResponse
from .models import Tutorial, Definitions
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from .forms import NewUserForm
from subprocess import run,PIPE
import sys
from django.views.decorators import gzip
from django.template import Context, Template
import cv2
import time
import os
import pickle #To save and retrieve weights
import tensorflow as tf #Deep learning library
import numpy as np#for Math
from music21 import instrument, chord, note, stream #to handle music files i.e save convert read etc
from tensorflow.keras.models import Sequential #to create a model
from tensorflow.keras.layers import Dense, Dropout, LSTM,Activation # to create and add these layers to model
import torch
import torchvision
import torch.nn as nn
import pylab
ROOT = os.path.abspath("main")
from PIL import Image 
sys.path.append('/Users/xenox/Documents/Coadddding/site/mysite/main')
sys.path.append('/Users/xenox/Documents/Coadddding/site/mysite/main/Gen_image/')
from gan import Discriminator, Generator


# Create your views here.

def First_page(request):
    return render(request = request,
                  template_name = 'main/First_page.html',
                  context = {'tutorials':Tutorial.objects.all})    

def homepage(request):
    return render(request = request,
                  template_name = 'main/home.html',
                  context = {'tutorials':Tutorial.objects.all})        


def register(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get("username")
            messages.success(request, f"New account created: {username}")
            login(request,user)
            messages.info(request,f"You are logged in as: {username}")
            return redirect("main:homepage")
        else:
            for msg in form.error_messages:
               messages.error(request, f"{msg}:{form.error_messages[msg]}")
    
    form = NewUserForm 
    return render(request = request,
                  template_name = "main/register.html",
                  context = {"form":form})

def logout_req(request):
    logout(request)
    messages.info(request, "Logged out successfully!!!")
    return redirect("main:homepage")


def login_req(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
           username = form.cleaned_data.get('username')
           password = form.cleaned_data.get('password')
           user = authenticate(username = username,password = password)
           if user is not None:
               login(request,user)
               messages.info(request,f"You are logged in as: {username}")
               return redirect("main:homepage")
           else:
               for msg in form.error_messages:
                   messages.error(request, f"{msg}:{form.error_messages[msg]}")
        else:
            messages.error(request, "invalid username or password")
    
    form = AuthenticationForm()
    return render(request,
                  "main/login.html",
                  {"form":form})
    
    
'''   
def obj_button(request):
 # out = run([sys.executable,'//Users//xenox//Documents//Coaaadinggg//site//mysite//mysite//main//static//main//file.py'],shell=False,stdout = PIPE)
  data = "Shubh"
  return render(request = request,
          template_name = 'main/obj_detection.html',
          context = {'data':data})       
'''

def face_recognition(request):
  return render(request = request,
                  template_name = 'main/face_recognition.html',
                  context = {'tutorials':Tutorial.objects.all})


face = cv2.CascadeClassifier('{}/Face recognition/data/haarcascade_frontalface_alt2.xml'.format(ROOT))
rec = cv2.face.LBPHFaceRecognizer_create()

font = cv2.FONT_HERSHEY_PLAIN

rec.read("{}/Face recognition/trained.yml".format(ROOT))
labels = {'name':1}

class VideoCamera_face(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        

    def __del__(self):
        self.video.release()

    def get_frame(self):
      
      with open('{}/Face recognition/label.pickle'.format(ROOT),'rb') as f:
        
        alabels = pickle.load(f)
        labels = {a:b for b,a in alabels.items()}

      ret, frame = self.video.read()
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
      faces = face.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 5)
      
      for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+h]
        
        idd,conf = rec.predict(roi)
        if conf >= 0.35:          
          cv2.putText(frame,labels[idd],(x,y + 20),font,1,(0,255,0),2)  
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


       
    # Display the resulting frame
      
      ret,jpeg = cv2.imencode('.jpg',frame)
      
      return jpeg.tobytes()  
       


def gen_face(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page


def run_face(request):
  if request.user.is_authenticated == True:
    try:
        return StreamingHttpResponse(gen_face(VideoCamera_face()),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        return "aborted"
  else:
    messages.error(request,"Please Login to use this feature")   

    return redirect("main:homepage")   
 

class VideoCamera_collect(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
  
    def __del__(self):
        self.video.release()

    def get_frame(self,name):
      label = name
      img_dir = "{}/Face recognition/Images".format(ROOT)
     
      if os.path.isdir(f'{ROOT}/Face recognition/Images/{label}'):
        print("WORKINNN")
      else:
          os.mkdir(f'{ROOT}/Face recognition/Images/{label}') 
          a = 1
          while(True):
            if a <= 20:
              ret,frame = self.video.read()
              gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
              faces = face.detectMultiScale(gray,scaleFactor = 1.5,minNeighbors = 5)

              for (x,y,w,h) in (faces):
                if (x,y,w,h) != (0,0,0,0) and a <= 20:
                  cv2.imwrite(f'{ROOT}/Face recognition/Images/{label}/{a}.jpg',frame)
                  a += 1
                  time.sleep(0.5)

                else:
                  break 
            else:
              time.sleep(1)
              break
         

          return "done"

def updating_weights_for_face_recognition(request):
  if request.user.is_authenticated == True:
    img_dir = "{}/Face recognition/Images/".format(ROOT)

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
          
          faces = face.detectMultiScale(img_array,scaleFactor = 1.5, minNeighbors = 5)

          for (x,y,w,h) in faces:
            region = img_array[y:y+h, x:x+w]
            x_train.append(region)
            y_train.append(id_)     

    with open('{}/Face recognition/label.pickle'.format(ROOT), 'wb') as f:
      pickle.dump(lab_ids, f)

    rec.train(x_train, np.array(y_train))
    rec.save("{}/Face recognition/trained.yml".format(ROOT))
    messages.success(request, "Model successfully trained now you can use Face Recognition")
    return render(request = request,
                    template_name = 'main/face_recognition.html',
                    context = {'tutorials':Tutorial.objects.all})
  else:
        messages.error(request, "Please Login to use this feature")
        return redirect("main:homepage")  
        

def gen_collection(camera,name):
  while True:
    frame = camera.get_frame(name = name)
    break
  return "done"


def collect(request):
  name = request.POST.get('FirstName', None)
  if request.user.is_authenticated == True:
    if name == "":
      messages.error(request, "Please enter something")
      return render(request = request,
                    template_name = 'main/face_recognition.html',
                    context = {'tutorials':Tutorial.objects.all})

    elif os.path.isdir(f'{ROOT}/Face recognition/Images/{name}'):
      messages.error(request, "Training has already been done for this person")
      return render(request = request,
                    template_name = 'main/face_recognition.html',
                    context = {'tutorials':Tutorial.objects.all})
    
    try:
      done = gen_collection(VideoCamera_collect(),name)
      print(done)
      messages.success(request,"Data successfully collected please TRAIN THE MODEL")
      
    except HttpResponseServerError as e:
      return "aborted"  
  else:
    messages.error(request, "Please Login to use this feature")
    return redirect("main:homepage")
  
  return render(request = request,
                    template_name = 'main/face_recognition.html',
                    context = {'tutorials':Tutorial.objects.all})        
        

def music_generation(request):
  return render(request = request,
                template_name = 'main/music_generation.html',
                context = {'tutorials':Tutorial.objects.all})

def face_generation(request):
  return render(request = request,
                  template_name = 'main/face_generation.html',
                  context = {'tutorials':Tutorial.objects.all})                                      

def gen_image():
  D = Discriminator()
  G = Generator()

# load weights
  D.load_state_dict(torch.load('{}/Gen_image/weights/weight_D.pth'.format(ROOT),map_location='cpu'))
  G.load_state_dict(torch.load('{}/Gen_image/weights/weight_G.pth'.format(ROOT),map_location='cpu'))

  batch_size = 25 #number of images generate in a batch
  latent_size = 100 

  noise = torch.randn(batch_size, latent_size, 1, 1)
  images = G(noise) 

  fake_images = images.cpu().detach().numpy()
  fake_images = fake_images.reshape(fake_images.shape[0], 3, 32, 32)
  fake_images = fake_images.transpose((0, 2, 3, 1))
  
  images = []
  for i in range(batch_size): 
    images.append(fake_images[i])

  vertical = np.vstack(tuple(images))
  vertical = ((vertical * vertical) * 255.0)/ (vertical.max() * vertical.max())
  cv2.imwrite("generated.jpeg", vertical)

#THIS PART OF THE CODE IS TAKEN FROM INTERNET
def download_img(request):
  file=f"{ROOT}/../generated.jpeg"
  f = open(file,"rb") 
  response = HttpResponse()
  response.write(f.read())
  response['Content-Type'] ='image/jpeg'
  response['Content-Length'] =os.path.getsize(file)
  return response

def gen_img(request):
  if request.user.is_authenticated == True:  
    gen_image()
    messages.success(request,"Image Generated successfully")
    return render(request = request,
                template_name = 'main/face_generation.html',
                context = {'tutorials':Tutorial.objects.all})

  else:
    messages.error(request,"Please Login to use this feature")
    return redirect("main:homepage")

def gen_m():
  ''' function to generate music'''
  #loading the notes files created while training
  with open('{}/music/data/notes'.format(ROOT), 'rb') as f: #opening the file as variable f
    notes = pickle.load(f) #Getting the notes in notes variable

  pitch_names = sorted(set(item for item in notes)) #sorting all the notes 
  n = len(set(notes)) #getting the total number of individual notes

  inp, norm_inp = normalize(notes, pitch_names, n) #getting input in categorical format and also the normalized input   

  model = make_model(norm_inp,n) #Getting the model from the funcion
  pred = gen_notes(model,inp,pitch_names,n)
  song = make_midi(pred)

def normalize(notes,pitch_names,n):
  #prepare the data for neural network
  #Also map the inputs to intigers to use in the network

  note_int = dict((note, number) for number, note in enumerate(pitch_names)) #Mapping numbers to notes and storing in the variabel

  l = 100 #min number of inuts req to make predictions
  inp = [] #to collect the input in integer form
  
  for i in range(0,len(notes) - l): #looping through all notes
    sequence_inp = notes[i:i+l] #taking the notes from i to i+l i.e 100 + i. to get the next notes
    seq_out = notes[i+l]  
    inp.append([note_int[x] for x in sequence_inp]) #getting the categorical number of hte note from note_int

  n_patterns = len(inp) #number of patterns
  
  norm_inp = np.reshape(inp, (n_patterns, l, 1)) #Reshaping the array to a tensor so that we can feed it in our network
  norm_inp = norm_inp / float(n) #normalizing the input 

  return (inp,norm_inp)
    
def make_model(inp,n):

  model = Sequential() #making object of sequential class
  model.add(LSTM(512, input_shape = (inp.shape[1], inp.shape[2]),return_sequences = True)) #adding a LSTM network and filling in the required parameters 
  model.add(Dropout(0.3)) #Addng dropout layer, regularization
  model.add(LSTM(512,return_sequences = True))
  model.add(Dropout(0.3))
  model.add(LSTM(512))
  model.add(Dense(256)) #basic deep neural network
  model.add(Dropout(0.3))
  model.add(Dense(n))
  model.add(Activation('softmax')) #to get the number of outputs same as number of individual notes
  model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop') #loss function for back-propogation

  model.load_weights('{}/music/weights.hdf5'.format(ROOT)) #loading the pre-trained weights to generate music

  return model

def gen_notes(model,inp,pitch_names,n):
  #pick a random seq from inp to start generation
  first = np.random.randint(0,len(inp) - 1)
  int_note = dict((number, note) for number, note in enumerate(pitch_names)) #getting the notes from number to convert the categorial integers back into notes

  start = inp[first] #get the first categorical note
  pred = [] #to store all the predictions

  for index in range(500): #to generate next 500 notes
    pred_inp = np.reshape(start,(1,len(start),1)) #changing into tensor
    pred_inp = pred_inp / float(n) #normalizing

    prediction = model.predict(pred_inp,verbose = 0) #to not display anything in console while  predcting next note from input

    most_likely = np.argmax(prediction) #get the max of all outputs
    res = int_note[most_likely] #get the note for the categorical variable
    pred.append(res) #adding the reslt to the final array of predictions

    start.append(most_likely) #adding the pred to start to continue the loop
    start = start[1:len(start)]  

  return pred 

#THIS PART OF THE CODE IS TAKEN FROM INTERNET
def make_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='main/music/test_output.mid')

def gen_music(request):
  if request.user.is_authenticated == True:  
    done = gen_m()
    messages.success(request,"Music Generated successfully")
    return render(request = request,
                template_name = 'main/music_generation.html',
                context = {'tutorials':Tutorial.objects.all})

  else:
    messages.error(request,"Please Login to use this feature")
    return redirect("main:homepage")

#THIS PART OF THE CODE IS TAKEN FROM INTERNET
def playAudioFile(request):
  file="main/music/test_output.mid"
  f = open(file,"rb") 
  response = HttpResponse()
  response.write(f.read())
  response['Content-Type'] ='audio/midi'
  response['Content-Length'] =os.path.getsize(file)
  return response

net = cv2.dnn.readNet('{}/yolov3.weights'.format(ROOT),'{}/yolov3.cfg'.format(ROOT)) # Using a module in cv2 to read the weights and cfg file
classes = [] #Intitizalizing the variable classes to store all the classes

with open("{}/coco.names".format(ROOT),"r") as f: # Opening the file containing the classes
  classes = [line.strip() for line in f.readlines()] #Storing the values of the classes in the array classes

layers = net.getLayerNames() #Getting all the layesrs of the Neural Network
output = [layers[i[0] -1] for i in net.getUnconnectedOutLayers()] # Extracting the last layer containing the output

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,image = self.video.read()
        h,w,c = image.shape 
        blob = cv2.dnn.blobFromImage(image,0.00392,(320,320),(0,0,0),True,crop = False)

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
                if confidence > 0.6:
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
        
        for i in range(total_objects):
          
            x,y,wid,hei = boxex[i]
            label = str(classes[class_ids[i]])
        
            cv2.rectangle(image,(x,y),(x+wid,y+hei),(255,0,0),2)
            cv2.putText(image,label,(x,y + 20),font,1,(0,255,0),2)
            cv2.resize(image,(300,300))
        ret,jpeg = cv2.imencode('.jpg',image)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page

def index(request): 
  if request.user.is_authenticated == True:  
    try:
        return StreamingHttpResponse(gen(VideoCamera()),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        return "aborted"
  else:
    messages.error(request,"Please Login to use this feature")
    return redirect("main:homepage")

def obj_detection(request):
  return render(request = request,
                  template_name = 'main/obj_detection.html',
                  context = {'tutorials':Tutorial.objects.all}) #add ,video: OUTPUT OF THTHON 

def extras(request):
  return render(request = request,
                template_name = 'main/Extras.html',
                context = {'definitions':Definitions.objects.all})

def projects(request):
  return render(request = request,
                template_name = 'main/projects.html',
                context = {'definitions':Definitions.objects.all})

def blog(request):
  return render(request = request,
                template_name = 'main/blog.html',
                context = {'definitions':Definitions.objects.all})

def notion(request):
  return render(request = request,
                template_name = 'main/notion.html',
                context = {'definitions':Definitions.objects.all})

import urllib.request as urllib2
import json

def videos(request):
  reqURL =  " https://api.rss2json.com/v1/api.json?rss_url=https%3A%2F%2Fwww.youtube.com%2Ffeeds%2Fvideos.xml%3Fchannel_id%3DUCDXQNb3linY9EksWAvJhPyQ&api_key=mll6zmdtx5s3w48ti7g1hdbrpjgzssl23q36c8ml&count=1000"
  response = urllib2.urlopen(reqURL)
  data = json.loads(response.read())
  value = len(data['items'])
  if value % 2 == 0:
    #total = [i for i in range(value)]
    i = [i for i in range(0,value,2)]
    j = [j for j in range(1,value,2)]
  else:
    #total = [i for i in range(value+1)]  
    i = [i for i in range(0,value+1,2)]
    j = [j for j in range(1,value+1,2)]

  print(i,j)
  return render(request = request,
                template_name = 'main/videos.html',
                context = {'list':zip(i,j)})                

