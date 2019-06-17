
# same aim, modify original code by any means except by applying filter. Thats the aim of the second program DetectWink2.py 



import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys
import math

def detectWink(frame, location, ROI, cascade): # frame is the image, cascade is eyesCascade
    # ROI = cv2.medianBlur(ROI, 1)
    # ROI = cv2.equalizeHist(ROI) 
    eyes = cascade.detectMultiScale( # can do histogram equalization/smoothing on ROI so as to improve detection
        ROI, 1.2, 6, 0|cv2.CASCADE_SCALE_IMAGE, (5, 5)) # again giving image, scaleFactor, minNeighbors, flag & minSize
    for e in eyes:
        e[0] += location[0] # getting location of eyes in the frame
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2) # draw a red rectangle of width 2 over each detected eye
    return len(eyes) == 1    # if number of eyes is one, return True

def detect(frame, faceCascade, eyesCascade): # frame is the image 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # converting color image to gray image. You may use open CV method to get L values directly

    # possible frame pre-processing to improve detection:*************************************
    # gray_frame = cv2.equalizeHist(gray_frame)  # histogram equalization
    gray_frame = cv2.medianBlur(gray_frame, 1) # median smoothing

    scaleFactor = 1.15 # range is from 1 to .. (never go below 1)
    minNeighbors = 3   # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 (changes template size - faster) or 0|cv2.CASCADE_SCALE_IMAGE (changes image size - more accurate)
    minSize = (10,10) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale( # returns a list of rectangles where faces were detected. detectMultiScale is a part of faceCascade
        gray_frame, 
        scaleFactor, # first tries by minSize rectangle, then scales it up by scaleFactor (larger => faster)
        minNeighbors, # if greater than or equal to minNeighbors (multiple rectangles) are detected, a face is present. If the same face in image is detected multiple times, increase this
        flag, 
        minSize) # if smaller faces were not detected, decrease minSize

    detected = 0
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3] # face-rectangle represented as a list of upper left had coordinates and width and height of rectangle  
        # print(w,h)
        faceROI = gray_frame[y:y+math.floor(0.6*h), x:x+w] # modify the ROI to concentrate on eye region only
        if detectWink(frame, (x, y), faceROI, eyesCascade): # calls another function to detect wink in the ROI, returns boolean
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2) # draw a blue rectangle of width 2 on the face if wink is detected
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2) # draw a green rectangle of width 2 on the face if wink is not detected
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))] # getting all image files from the folder

    windowName = None
    totalCount = 0
    for f in files: # goes through each image in the folder
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)  # THE DETECTION PART - see detect function
            totalCount += lCnt # each image can have more than one person, so lCnt >= 0 # total winks detected
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0) # waits for key press to go to the next image in folder 
    return totalCount  # returns the total count of winks in the folder as a whole

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0: # loop until key pressed for 30 secs
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":                                            # main function
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2: # need 0 (live detection) or 1 argument(s) (detection from images folder)
        print(sys.argv[0] + ": got " + len(sys.argv) - 1
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pre-trained cascades from openCV for face and eye detection respectively
	
	# "other" classifier (.xml) options: eyes: haarcascade_eye_tree_eyeglasses
	# face: haarcascade_frontalcatface, haarcascade_frontalcatface_extended, haarcascade_frontalface_alt (good), haarcascade_frontalface_alt_tree, haarcascade_frontalface_alt2 (best).
	# haarcascade_eye_custom1 to 3, haarcascade_face_custom1 to 5 (4 is good)
	
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')

    if(len(sys.argv) == 2): # one argument => face and wink detection from the images folder
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments => face and wink detection from live video capture
        runonVideo(face_cascade, eye_cascade)

