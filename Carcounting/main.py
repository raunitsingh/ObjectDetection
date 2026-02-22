import cv2
from pysimverse import Drone

# Initialization
drone = Drone()
drone.connect()
drone.streamon()

# Image Capture
while True:
    img, _ = drone.get_frame()
    cv2.imshow("Image", img)  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
