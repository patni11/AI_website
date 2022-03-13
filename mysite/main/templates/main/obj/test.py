import obj_copy
import cv2

while True:
	frame = obj_copy.detect_obj()
	cv2.imshow("img",frame)
	key = cv2.waitKey(1)
	if key == 27:
		break

cv2.destroyAllWindows()		

	

    	