from ultralytics import YOLO
import cv2
import cvzone
from sort import Sort  # Make sure sort.py is in your folder
import math

model = YOLO("yolov8n.pt")
TARGET_CLASS = 0  # COCO person

# Load video
video_path = "recording2.avi"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("processed_sorted.avi", fourcc, fps, (frame_width, frame_height))

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Keep count of unique IDs
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = []

    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == TARGET_CLASS and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

    # Convert to numpy array for SORT
    import numpy as np
    if len(detections) > 0:
        dets_np = np.array(detections)
    else:
        dets_np = np.empty((0, 5))

    # Update SORT tracker
    tracked = tracker.update(dets_np)

    for track in tracked:
        x1, y1, x2, y2, track_id = map(int, track)
        w, h = x2 - x1, y2 - y1
        unique_ids.add(track_id)

        cvzone.cornerRect(frame, (x1, y1, w, h))
        cvzone.putTextRect(frame, f'ID {track_id}', (x1, y1 - 10), scale=1, thickness=1)

    out.write(frame)
    cv2.imshow("YOLOv8 + SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Total unique persons detected: {len(unique_ids)}")
