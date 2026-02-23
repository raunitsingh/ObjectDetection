import cv2
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

model = UltralyticsDetectionModel(
    model_path="models/yolo11m.pt",
    confidence_threshold=0.5,   #  LOWER for aerial humans
    device="cuda"                # cpu if no GPU
)

cap = cv2.VideoCapture("yt6.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # SAHI sliced detection (NO resize here)
    result = get_sliced_prediction(
        frame,
        model,
        slice_height=1280,          #  smaller slices
        slice_width=1280,
        overlap_height_ratio=0.3,  #  higher overlap
        overlap_width_ratio=0.3,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
    )

    for obj in result.object_prediction_list:

        # 👤 PERSON ONLY
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

    # resize ONLY for display
    display = cv2.resize(frame, None, fx=0.6, fy=0.6)
    cv2.imshow("Frame", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
