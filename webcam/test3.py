import cv2
from sahi.models.ultralytics import UltralyticsDetectionModel


sahi_od = UltralyticsDetectionModel("models/yolo11m.pt")


cap = cv2.VideoCapture("yt7.mp4")



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


    frame = cv2.resize(frame, dsize= None, fx=0.5, fy=0.5)
    cv2.imshow("Frame", frame)
    key =  cv2.waitKey(1) 
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()