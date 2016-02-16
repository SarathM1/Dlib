import cv2

#Load a cascade file for detecting faces
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')	# Provide correct path to faces.xml

#to use video file instead of camera
#cap = cv2.VideoCapture('openCV2.mp4')	#use correct path or puth the video in same folder

#To use live feed from camera
#cap = cv2.VideoCapture(0)

#while True:

# read frames from captured video
#ret,frame = cap.read()

frame = cv2.imread('test.jpg')
print frame
#Convert to grayscale
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#Look for faces in the image using the loaded cascade file
faces = faceCascade.detectMultiScale(gray,1.1,5)

#Draw a rectangle around every found face
for (x,y,w,h) in faces:
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

# show resulting video(played frame by frame) with rectangles around detected face	
cv2.imshow('frame',frame)

#use waitKey(1) for live feed from camera
# Press 'q' to stop program execution
#if cv2.waitKey(20) & 0xff == ord('q'):
#break
cv2.waitKey()
#Stop capture
#cap.release()

# Properly close all windows
cv2.destroyAllWindows()