import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int (bboxC.width * iw), int(bboxC.height * ih)
                
                bboxs.append([id, bbox, detection.score])
                img = self.fancyDraw(img, bbox)
                cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        
        cv.rectangle(img, bbox, (255,0,255), rt)
        # Top Left x, y
        cv.line(img, (x, y), (x+l, y), (255,0,255), t)
        cv.line(img, (x, y), (x, y+l), (255,0,255), t)

        # Top Right x, y
        cv.line(img, (x1, y), (x1-l, y), (255,0,255), t)
        cv.line(img, (x1, y), (x1, y+l), (255,0,255), t)

        # Bottom Left x, y
        cv.line(img, (x, y1), (x+l, y1), (255,0,255), t)
        cv.line(img, (x, y1), (x, y1-l), (255,0,255), t)

        # Bottom Right x, y
        cv.line(img, (x1, y1), (x1-l, y1), (255,0,255), t)
        cv.line(img, (x1, y1), (x1, y1-l), (255,0,255), t)

        return img
def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
        cv.imshow("Image", img)
        cv.waitKey(1)

if __name__ == '__main__':
    main()