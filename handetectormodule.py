import cv2
import  mediapipe as mp
import time

cap=cv2.VideoCapture(1)

class handDetector():
    def __init__(self,mode=False,maxhands=2,detectionCon=0.5,trackCon=0.5):
       self.mode=mode
       self.maxhands=maxhands
       self.detectionCon=detectionCon
       self.trackCon=trackCon


       self.mphands=mp.solutions.hands
       self.hands=self.mphands.Hands(self.mode,self.maxhands,self.detectionCon,self.trackCon)
       self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img,draw=True):

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                 self.mpDraw.draw_landmarks(img,handLms,self.mphands.HAND_CONNECTIONS)

        return img
    def findposition(self,img,handno=0,draw=True):
        LMlist=[]
        if self.results.multi_hand_landmarks:
                myhand= self.results.multi_hand_landmarks[handno]
                for id, lm in enumerate(myhand.landmark):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id, cx, cy)
                    LMlist.append(id,cx,cy)
                    if draw:
                     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return LMlist
    #cv2.putText(img,str(int(fps)),(18,78),3,(255,8,255),3)




def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(1)
    detector=handDetector()

    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        LMList=detector.findposition(img)
        if len(LMList)!=0:
            print(LMList[4])

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (18, 78), 3, (255, 8, 255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()