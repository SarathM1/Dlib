import dlib,numpy
from skimage import io
import cv2
from math import sqrt

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [MOUTH_POINTS]
FEATHER_AMOUNT = 11


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

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
	img = cv2.imread('test.jpg')
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
	
	if(eccentricity < 0.84):
		cv2.putText(img,'Commands = O',(10,450), font, 1,(0,0,255),2,16)
	else:
		cv2.putText(img,'Commands = E',(10,500), font, 1,(0,0,255),2,16)
		
	cv2.imwrite('output_test.jpg',img)

	cv2.imshow('Mask',img_copy)
	cv2.imshow('Output', output_img)
	cv2.imshow('Img',img)

	cv2.waitKey()

if __name__ == '__main__':
	try:
		main()
	except ValueError as e:
		print e
	
