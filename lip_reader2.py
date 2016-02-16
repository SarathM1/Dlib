# import the necessary packages
import dlib,numpy
from skimage import io

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils
from math import sqrt

print 1
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [MOUTH_POINTS]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
print 2
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (720, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(720, 480))
print 3
# allow the camera to warmup
time.sleep(0.1)

def get_landmarks(im):
    rects = detector(im, 0)
    
    if len(rects) > 1:
        print 'TooManyFaces'
    if len(rects) == 0:
    	raise ValueError('Error: NoFaces!!')

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    for group in OVERLAY_POINTS:
        hull = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(im, hull, 0) 

def main():
	print "\tSTARTING!!"
	# capture frames from the camera
	for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
		# grab the raw NumPy array representing the image, then initialize the timestamp
		# and occupied/unoccupied text
		img= frame.array
		img = imutils.resize(img, width = 200)
		img_copy = img.copy()
		cv2.imwrite('lip_reader2.jpg',img)
		landmarks = get_landmarks(img)

		get_face_mask(img_copy, landmarks)
		output_img = img-img_copy

		output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
		contours,hierarchy = cv2.findContours(output_img.copy(), cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)  #cv2.findContours(image, mode, method
		#cv2.drawContours(img, contours, -1, (0,255,0), 2,maxLevel=0)
		
		cnt = contours[0]
		ellipse = cv2.fitEllipse(cnt)
		(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
		cv2.ellipse(img,ellipse,(0,255,0),2)

		
		a = ma/2
		b = MA/2


		eccentricity = sqrt(pow(a,2)-pow(b,2))
		eccentricity = round(eccentricity/a,2)

		font = cv2.FONT_HERSHEY_SIMPLEX

		#cv2.putText(img,'ma = '+str(round(ma,2)),(10,100), font, 1,(255,0,0),1,1)
		#cv2.putText(img,'MA = '+str(round(MA,2)),(10,150), font, 1,(255,0,0),1,1)
		cv2.putText(img,'Eccentricity = '+str(round(eccentricity,3)),(10,100), font, 1,(255,0,0),1,1)
		
		if(eccentricity < 0.84):
			print 'O'
			cv2.putText(img,'Commands = O',(10,150), font, 1,(0,0,255),1,1)
		else:
			print 'E'
			cv2.putText(img,'Commands = E',(10,150), font, 1,(0,0,255),1,1)
			
		#cv2.imwrite('output_e3.jpg',img)

		cv2.imshow('Mask',img_copy)
		cv2.imshow('Output', output_img)
		cv2.imshow('Img',img)
		# show the frame
		key = cv2.waitKey(1) & 0xFF

		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
		    break# import the necessary packages

if __name__ == '__main__':
	try:
		main()
	except ValueError as e:
		print e
