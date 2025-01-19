import os
import cv2 as cv

dir="./data"
if not os.path.exists(dir):
    os.makedirs(dir)

number_of_classes=26
images=500

cam=cv.VideoCapture(0)
for i in range(number_of_classes):
    if not os.path.exists(os.path.join(dir, str(i))):
        os.makedirs(os.path.join(dir, str(i)))
    print ("Class {}".format(str(i)))
    while(True):
        bool, frame=cam.read()
        cv.putText(frame, "Press a to capture", (100,100), cv.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2, cv.LINE_AA)
        cv.imshow("frame", frame)
        if cv.waitKey(20)==ord("a"):
            break
    counter=0
    while counter<images:
        bool, frame=cam.read()
        cv.imshow("frame", frame)
        cv.waitKey(20)
        cv.imwrite(os.path.join(dir,str(i),'{}.jpg'.format(counter)), frame)
        counter += 1

cam.release()
cv.destroyAllWindows()