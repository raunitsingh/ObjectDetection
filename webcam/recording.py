from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Make sure this is in your project folder or give full path

# Webcam setup
cap = cv2.VideoCapture("yt2.mp4")  # Change to 0 for webcam
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20.0

# Setup video writer
output_path = "person_detection.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Only detect 'person' (class 0 in COCO)
TARGET_CLASS = 0
CONFIDENCE_THRESHOLD = 0.5

print("[INFO] Recording started. Press 'q' to stop.")

while True:
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to read from webcam.")
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == TARGET_CLASS and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f'Person {conf:.2f}'

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Show frame and write to video
    cv2.imshow("YOLOv8 Person Detection", frame)
    out.write(frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Recording stopped.")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
