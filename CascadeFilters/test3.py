import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import imutils

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 8
rawCapture = PiRGBArray(camera, size=(640, 480))
print 3
# allow the camera to warmup
time.sleep(0.1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def main():
	print "\tSTARTING!!"
	# capture frames from the camera
	for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
		# grab the raw NumPy array representing the image, then initialize the timestamp
		# and occupied/unoccupied text
		img= frame.array
		#img = imutils.resize(img, width = 500)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		faces = face_cascade.detectMultiScale(gray,1.1,5)
		for (x,y,w,h) in faces:
		    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		    roi_gray = gray[y:y+h, x:x+w]
		    roi_color = img[y:y+h, x:x+w]
		    eyes = eye_cascade.detectMultiScale(roi_gray)
		    for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		cv2.imshow('img',img)
		key = cv2.waitKey(1) & 0xFF

		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
		    break# import the necessary packages
		

if __name__ == '__main__':
	try:
		main()
		cv2.destroyAllWindows()
	except ValueError as e:
		print e
