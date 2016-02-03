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
    #im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,landmarks[group],color=1)

    #im = numpy.array([im, im, im]).transpose((1, 2, 0))

    #im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    #im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

img = cv2.imread('test.jpg')
landmarks = get_landmarks(img)

mask = get_face_mask(img, landmarks)

#cv2.bitwise_and(img,img,mask = mask)

cv2.imshow('Mask',img)
cv2.waitKey()

#warped_mask = warp_im(mask, M, im1.shape)
#combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],axis=0)
