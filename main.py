import cv2
import cvzone
import numpy as np 
from FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
textList = ["Hi, My Name is", "Shubharthak","Sangharasha.",
            "I am studying in", "Chandigarh University",
            "Currently in second year"]
sen = 10 #more is less sensitive
while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img,draw=False)
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        #Drawing
        # cv2.line(img,pointLeft,pointRight,(0,200,0),3)
        # cv2.circle(img,pointLeft,5,(255,0,255),cv2.FILLED)
        # cv2.circle(img,pointRight,5,(255,0,255),cv2.FILLED)
        w,_ = detector.findDistance(pointLeft,pointRight)
        W = 6.3 #average in cm 
        #finding focal length 
        # d = 50 
        # f = (w * d)/ W        
        # Finding distance 
        f = 700
        d = (W * f) / w
        cvzone.putTextRect(img,f'Depth: {int(d)}cm',(face[10][0]-100, face[10][1]-50), scale = 2)
        
        for i, text in enumerate(textList):
            singleHeight = 20 + int((int(d/sen)*sen)/4)
            scale = 0.4 + (int(d/sen)*sen)/75
            cv2.putText(imgText, text, (50,50+(i*singleHeight)),cv2.FONT_ITALIC,scale,(255,255,255),2)

                
    imgStacked = cvzone.stackImages([img, imgText],2,1)
    cv2.imshow('Face Depth', imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()