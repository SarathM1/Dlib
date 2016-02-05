import dlib,numpy
from skimage import io
import cv2
from math import sqrt
import numpy as np


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [MOUTH_POINTS]
FEATHER_AMOUNT = 11


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cam = cv2.VideoCapture(0)

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

def hand_track(img):
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
			
			if far[1] >= cy:
				continue
			else:
				pass
				#print far[1],cy
			#dist = cv2.pointPolygonTest(cnt,far,True)
			cv2.circle(img,far,8,[0,0,255],-1)
			prev_start = start
			cntr +=1

		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,"No of finger's = "+str(cntr+1),(10,250), font, 1,(0,255,0),2,16)

	try:
		cv2.imshow('HSV',drawing)
	except Exception, e:
		pass	
	
	cv2.imshow('Threshold',thresh)

def main():
	while True:
		try:
			ret,img = cam.read()
			hand_track(img)
			if not ret:
				print "Error opening file"
				break
			img_copy = img.copy()
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

			cv2.putText(img,'ma = '+str(round(ma,2)),(10,300), font, 1,(255,0,0),2,16)
			cv2.putText(img,'MA = '+str(round(MA,2)),(10,350), font, 1,(255,0,0),2,16)
			cv2.putText(img,'Eccentricity = '+str(round(eccentricity,3)),(10,400), font, 1,(255,0,0),2,16)
			
			if(eccentricity < 0.88):
				cv2.putText(img,'Commands = O',(10,450), font, 1,(0,0,255),2,16)
			else:
				cv2.putText(img,'Commands = E',(10,450), font, 1,(0,0,255),2,16)
				
			cv2.imshow('Mask',img_copy)
			cv2.imshow('Output', output_img)
			cv2.imshow('Img',img)

			if cv2.waitKey(20) & 0xFF == ord('q'):	# To move frame by frame
					print "Pressed Q, quitting!!"
					break
		except ValueError as e:
			print e
if __name__ == '__main__':
	main()
