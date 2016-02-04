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
    rects = detector(im, 1)
    
    if len(rects) > 1:
        print 'TooManyFaces'
    if len(rects) == 0:
        print 'NoFaces'

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    for group in OVERLAY_POINTS:
        hull = cv2.convexHull(landmarks[group])
        cv2.fillConvexPoly(im, hull, 0) 

img = cv2.imread('o3.jpg')
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

leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

diff1 = abs(leftmost[0] - rightmost[0])
diff2 = abs(topmost[1] - bottommost[1])

a = diff1/2
b = diff2/2
a2 = ma/2
b2 = MA/2

eccentricity = sqrt(pow(a,2)-pow(b,2))
eccentricity = round(eccentricity/a,2)

eccentricity2 = sqrt(pow(a2,2)-pow(b2,2))
eccentricity2 = round(eccentricity2/a2,2)

font = cv2.FONT_HERSHEY_SIMPLEX

"""
cv2.putText(img,'Leftmost = '+str(leftmost),(10,100), font, 1,(255,0,0),2,16)
cv2.putText(img,'Rightmost = '+str(rightmost),(10,150), font, 1,(255,0,0),2,16)
cv2.line(img,leftmost,rightmost,(255,0,0),2)

cv2.putText(img,'Topmost = '+str(topmost),(10,200), font, 1,(255,0,0),2,16)
cv2.putText(img,'Bottommost = '+str(bottommost),(10,250), font, 1,(255,0,0),2,16)
cv2.line(img,topmost,bottommost,(255,0,0),2)
"""
cv2.putText(img,'Diff1 = '+str(diff1)+','+str(round(ma,2)),(10,300), font, 1,(255,0,0),2,16)
cv2.putText(img,'Diff2 = '+str(diff2)+','+str(round(MA,2)),(10,350), font, 1,(255,0,0),2,16)
cv2.putText(img,'Eccentricity = '+str(eccentricity)+','+str(eccentricity2),(10,400), font, 1,(255,0,0),2,16)

cv2.imwrite('output_o3.jpg',img)


cv2.imshow('Mask',img_copy)
cv2.imshow('Output', output_img)
cv2.imshow('Img',img)

cv2.waitKey()

