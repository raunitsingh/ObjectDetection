import cv2
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# Load model
model = UltralyticsDetectionModel(
    model_path="models/yolo11m.pt",
    confidence_threshold=0.15,
    device="cuda"  # change to "cpu" if no GPU
)

cap = cv2.VideoCapture("yt7.mp4")

frame_id = 0
DETECT_EVERY_N_FRAMES = 2  # increase to 3 if still slow

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Resize frame BEFORE detection (speed boost)
    frame = cv2.resize(frame, None, fx=0.6, fy=0.6)

    # Skip frames for speed
    if frame_id % DETECT_EVERY_N_FRAMES != 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    # SAHI sliced inference
    result = get_sliced_prediction(
        frame,
        model,
        slice_height=384,
        slice_width=384,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    for obj in result.object_prediction_list:

        # 👤 ONLY PERSON CLASS (COCO class 0)
        if obj.category.id != 0:
            continue

        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        score = obj.score.value

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
