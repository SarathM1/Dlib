import cv2,time
import numpy as np

cam = cv2.VideoCapture(0)

def main():
	while True:
		ret,img = cam.read()
		if not ret:
			print "Error opening file"
			break
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
			defects = cv2.convexityDefects(cnt,hull)
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
				if d<10000:
					continue
				
				if far[1] >= cy:
					continue
				else:
					pass
					#print far[1],cy
				dist = cv2.pointPolygonTest(cnt,far,True)
				cv2.circle(img,far,8,[0,0,255],-1)
				prev_start = start
				cntr +=1

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(img,"No of finger's = "+str(cntr),(10,300), font, 1,(0,255,0),2,16)

			
			cv2.imshow('HSV',drawing)
			cv2.imshow('Threshold',thresh)
			cv2.imshow('BGR',img)
			
			if cv2.waitKey(20) & 0xFF == ord('q'):	# To move frame by frame
				print "Pressed Q, quitting!!"
				break

if __name__ == '__main__':
	time.sleep(1)
	main()