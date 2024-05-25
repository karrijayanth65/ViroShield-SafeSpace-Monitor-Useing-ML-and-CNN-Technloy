# ViroShield-SafeSpace-Monitor-Useing-ML-and-CNN-Technology
An AI and IoT-driven solution to enforce COVID-19 protocols. Features mask detection using Convolutional Neural Networks (CNNs) and temperature monitoring with advanced sensors. Provides real-time email and audio/visual alerts, integrated via Raspberry Pi. Utilizes NumPy, Pandas, Matplotlib, and Plotly for robust data management and visualization.

#pyCode

     import cv2
     import numpy as np
     import datetime
     import subprocess
     from pydub import AudioSegment
     import time
     import sys
     sys.path.append('/home/pi/Templates')
     import mask_pred
     net = cv2.dnn.readNet('training_final.weights', 'testing.cfg')
     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

 
     classes = []
       with open("classes.txt", "r") as f:
       classes = f.read().splitlines()

     while True:
         g = datetime.datetime.now()
         subprocess.Popen("sudo fswebcam image.jpg",shell=True).communicate()
         img = cv2.imread('/home/pi/Mask_Detection/image.jpg')
         height, width, _ = img.shape 
         font = cv2.FONT_HERSHEY_PLAIN
         blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
         net.setInput(blob)
         output_layers_names = net.getUnconnectedOutLayersNames()
         layerOutputs = net.forward(output_layers_names)
     boxes = []
    confidences = []
    class_ids = []

         for output in layerOutputs:
         for detection in output:
                 scores = detection[5:]
                 class_id = np.argmax(scores)
                 confidence = scores[class_id]
                      if confidence > 0.2:
                          center_x = int(detection[0]*width)
                          center_y = int(detection[1]*height)
                      w = int(detection[2]*width)
                      h = int(detection[3]*height)

                          x = int(center_x - w/2)
                          y = int(center_y - h/2)

                     boxes.append([x, y, w, h])
                     confidences.append((float(confidence)))
                     class_ids.append(class_id)

          indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
          img, label = mask_pred.mask(indexes,x, y, w, h,boxes,classes,confidences,class_ids,img,g,font)
       cv2.imshow('Image', img)
           key = cv2.waitKey(1000)
            time.sleep(3)
         if(label=="MASK_NOT_FOUND"):
             audio = AudioSegment.from_mp3("/home/pi/Mask_Detection/Telugu.mp3")
             subprocess.call(["ffplay", "-nodisp", "-autoexit", '/home/pi/Mask_Detection/Telugu.mp3'])
             audio = AudioSegment.from_mp3("/home/pi/Mask_Detection/English.mp3")
             subprocess.call(["ffplay", "-nodisp", "-autoexit", '/home/pi/Mask_Detection/English.mp3'])
       key = cv2.waitKey(1000)
       cv2.destroyAllWindows()
