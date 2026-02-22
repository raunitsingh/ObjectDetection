from ultralytics import YOLO
import cv2
import cvzone
import math
from datetime import datetime
import os


model = YOLO("yolov8n.pt")  


TARGET_CLASS = 0

# Input video
video_path = "yt2.mp4"  
cap = cv2.VideoCapture(video_path)

# output video 
output_path = "processed_persons.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi or 'mp4v' for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Setup for unique detections
person_count = 0
detected_coords = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Avoid duplicate bounding box logging
                bbox = (x1, y1, x2, y2)
                if bbox not in detected_coords:
                    detected_coords.append(bbox)
                    person_count += 1
                    print(f" Person #{person_count} detected at ({x1},{y1}) - Confidence: {conf}")

                # Draw detection box
                cvzone.cornerRect(frame, (x1, y1, w, h))
                cvzone.putTextRect(frame, f'Person {conf}', (x1, y1 - 10), scale=1, thickness=1)

    # Show live video
    cv2.imshow("Person Detection - YOLOv8n", frame)
    out.write(frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()


print(f" Total unique person boxes detected: {person_count}")
