from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")  

# video capture
cap = cv2.VideoCapture("yt6.mp4")

# Set frame dimensions
cap.set(3, 1280)
cap.set(4, 720)

# Target class (person) 
TARGET_CLASS = 0  # COCO class index for "person"
CONFIDENCE_THRESHOLD = 0.1

while True:
    success, frame = cap.read()
    if not success:
        print("Video ended or failed to load frame.")
        break

    # Detect with YOLOv8
    results = model(frame, stream=True)

    # Draw bounding boxes for persons with confidence > 0.5
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == TARGET_CLASS and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]

                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Person Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()