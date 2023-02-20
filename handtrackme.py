import cv2
import  mediapipe as mp
import time

cap=cv2.VideoCapture(1)

mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw=mp.solutions.drawing_utils

ctime=0
ptime=0
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
           for id,lm in enumerate (handLms.landmark):
               #print(id,lm)
               h,w,c=img.shape
               cx,cy=int(lm.x*w),int(lm.y*h)
               print(id,cx,cy)
               cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

           mpDraw.draw_landmarks(img,handLms,mphands.HAND_CONNECTIONS)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime = ctime

    #cv2.putText(img,str(int(fps)),(18,78),3,(255,8,255),3)

    cv2.imshow("image",img)
    cv2.waitKey(1)

