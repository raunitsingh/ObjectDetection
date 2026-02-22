from ultralytics import YOLO
import cv2
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction

##from engine.sahi_object_detection import SAHIObjectDetection


sahi_od = SAHIObjectDetection("models/yolov8s.pt")



MODEL_PATH = "yolov8s.pt"         
IMG_SIZE = 1280                 # IMPORTANT for small objects
## CONF_THRESHOLD = 0.1            # Low confidence for aerial view
TARGET_CLASS = 0                # person
##TILE_SIZE = 640                 # tile size for frame splitting
##OVERLAP = 0.4                   # tile overlap


model = YOLO("yolov8s.pt")


cap = cv2.VideoCapture("yt7.mp4")
if not cap.isOpened():
    raise IOError(" Could not open video")

print(" Video loaded successfully")

 


while True:
    ret, frame = cap.read()
    if not ret:
        break



    predictions = sahi_od.detect(frame)
    for pred in predictions:
        bbox = pred.bbox
        class_id = pred.category.id
        score = pred.score.value


        x, y, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        cv2.rectangle(frame, (x, y), (x2, y2), sahi_od.colors[class_id], 3)




    frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    cv2.imshow(winname="Frame", mat=frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
