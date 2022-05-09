import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv.VideoCapture('Videos/1.mp4')
pTime = 0
while True:
    success, img = cap.read()
    resized = cv.resize(img, (1280,720), interpolation= cv.INTER_AREA)
    imgRGB = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(resized, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = resized.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(resized, (cx, cy), 10, (255,0,0), cv.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv.putText(resized, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv.imshow("Image",resized)

    cv.waitKey(1)