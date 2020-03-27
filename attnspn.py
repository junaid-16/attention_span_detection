from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
from imutils import resize
import time
import dlib
import cv2


'''The eye aspect ratio will be approximately constant when the eye is open.
The value will then rapid decrease towards zero during a blink.

If the eye is closed, the eye aspect ratio will again remain approximately constant,
but will be much smaller than the ratio when the eye is open'''

def eye_aspect_ratio(eye):
    #compute euclidean distance between the two sets of vertical
    #eye landmarks(x,y) coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    #compute euclidean distance between the horizontal
    #eye landmark (x,y) coordinates
    C = dist.euclidean(eye[0], eye[3])

    #compute eye aspect ratio
    EAR = (A + B)/(2.0 * C)

    return EAR

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required = True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

EAR_THRESH = 0.2 #assumed value(junaid), tuning might be required
EAR_CONSEC_FRAMES = 50 #again assumed by me ;-)


E_COUNTER = 0 #frame counter for eye

#using dlib's face detector (histogram of oriented gradient based) and then create
#the facial landmark predictor kyuki dlib me aisach hota hai
print("[INFO] loading facial landmark detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#grabing the indexes of facial landmarks for both eyes
(lStart,lEnd)= face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd)= face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#loop over frames
while True:
    # grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
    frame = vs.read()
    frame = resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces in the grayscale frame
    faces = detector(gray, 0)

    for face in faces:
        #taking facial landmark by using dlib facial landmark detector for face region
        #then convert the landmark coordinates to a numpy array
        shape = predictor(gray , face)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        #average of EAR of both eye
        ear = (leftEAR + rightEAR) / 2.0

        #compute the outline(hull) for eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1)  #-1 because we want to draw all the counter and we dont have any index for it
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0),1)


        #threshhold logic for eyes
        if ear < EAR_THRESH:
            E_COUNTER +=1

            #check for counter whether it have surpassed the threshhold
            if E_COUNTER >= EAR_CONSEC_FRAMES:
                #put a text on the frames
                cv2.putText(frame , "! attention ", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

        else:
            E_COUNTER = 0 #reseting eye counter

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0,255),2)

    cv2.putText(frame, "no of faces: {}".format(len(faces)), (150,15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow("Frame" ,frame)
    key = cv2.waitKey(1) & 0xFF
    if key ==ord("q"):
        break

cv2.destroyAllWindows()
vs.stop
