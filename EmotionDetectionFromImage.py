from deepface import DeepFace
import cv2 as cv
import matplotlib.pyplot as plt
import time



face_cascade= cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")
photo= cv.VideoCapture(0)
while(1):
   start_time = time.time()
   #Get photo from cam
   ret, frame=photo.read()
   #Face Detection and Analyze
   faces=face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
   for (x,y,w,h) in faces:
      cv.rectangle(frame,(x,y),(x+w,y+h),(0,0.255),1)
      emotions = DeepFace.analyze(frame,["emotion"])
      dominant_emotion=max(emotions[0]["emotion"],key=emotions[0]["emotion"].get)
      cv.putText(frame,f"emotion:{dominant_emotion}",(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
   end_time = time.time()
   elapsed_time = end_time-start_time
   print("işlem süresi:",elapsed_time)
   cv.imshow("photo",frame)
   k=cv.waitKey(0) & 0xFF
   if k==27:
      break
cv.destroyAllWindows()   
