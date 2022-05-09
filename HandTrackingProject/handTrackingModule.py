import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)


        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c =img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv.circle(img, (cx, cy), 15, (255,0,255), cv.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []
        if len(self.lmList):
            # Thumb
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)  

            # 4 Fingers        
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        else:
            fingers = [0, 0, 0, 0, 0]
        return fingers

def main():
    pTime = 0 # Previous Time
    cTime = 0 # Current Time
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime= time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)),(10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)
if __name__ == "__main__":
    main()
