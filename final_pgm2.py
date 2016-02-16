# import the necessary packages

from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import imutils

import dlib,numpy
from skimage import io

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils
from math import sqrt

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)



print 1
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [MOUTH_POINTS]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
print 2


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

def finger_count(img):
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

	# define range of blue color in HSV
	lower = np.array([20,100,100])
	upper = np.array([40,255,255])
	
	mask = cv2.inRange(hsv, lower, upper)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(img,img, mask= mask)
	res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(res,0,255,cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(thresh.copy(),cv2.cv.CV_RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	max_area = 0
	ci = 0
	defects = None
	for i in range(len(contours)):
		cnt=contours[i]
		area = cv2.contourArea(cnt)
		if(area>max_area):
			max_area=area
			ci=i

	if ci != 0:
		cnt=contours[ci]
		M = cv2.moments(cnt)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		cv2.circle(img,(cx,cy),8,[255,0,0],-1)

		hull = cv2.convexHull(cnt)
		drawing = np.zeros(img.shape,np.uint8)
		cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
		cv2.drawContours(drawing,[hull],0,(0,0,255),2)
		hull = cv2.convexHull(cnt,returnPoints = False)

		try:
			defects = cv2.convexityDefects(cnt,hull)
		except Exception, e:
			print e
		
		mind=0
		maxd=0
		i=0
		
		prev_start = (0,0)
		cntr = 0
		

	if defects is not None:
		for i in range(defects.shape[0]):
			s,e,f,d = defects[i,0]
			start = tuple(cnt[s][0])
			end = tuple(cnt[e][0])
			far = tuple(cnt[f][0])
			if d<12000:
				continue
			
			if far[1] >= (cy+20):
				continue
			else:
				pass
				#print far[1],cy
			#dist = cv2.pointPolygonTest(cnt,far,True)
			cv2.circle(img,far,8,[0,0,255],-1)
			prev_start = start
			cntr +=1

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,"No of finger's = "+str(cntr+1),(10,300), font, 1,(0,255,0),2,16)

	try:
		cv2.imshow('HSV',drawing)
	except Exception, e:
		pass	
	
	cv2.imshow('Threshold',thresh)
	cv2.imshow('BGR',img)

	cv2.imwrite('cam_video2.jpg',img)



def lip_reader(img):
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


def main():
	print "\tSTARTING!!"
	# capture frames from the camera
	for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
		# grab the raw NumPy array representing the image, then initialize the timestamp
		# and occupied/unoccupied text
		img= frame.array
		img = imutils.resize(img, width = 200)
		finger_count(img.copy())
		lip_reader(img.copy())
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
