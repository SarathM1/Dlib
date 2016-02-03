import dlib,numpy
from skimage import io
import cv2

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

img = cv2.imread('e.jpg')
img_copy = img.copy()
landmarks = get_landmarks(img)

get_face_mask(img_copy, landmarks)
output_img = img-img_copy

output_img = cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
contours,hierarchy = cv2.findContours(output_img.copy(), cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE)  #cv2.findContours(image, mode, method
cv2.drawContours(img, contours, -1, (0,255,0), 2,maxLevel=0)
cnt = contours[0]
M = cv2.moments(cnt)
for each_val in M:
	print each_val, ' : ',M[each_val]

cv2.imwrite('output_e.jpg',output_img)

cv2.imshow('Img',img)
cv2.imshow('Mask',img_copy)
cv2.imshow('Output', output_img)

cv2.waitKey()

