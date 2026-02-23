import cv2
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

# -----------------------------
# Load SAHI + YOLO model
# -----------------------------
model = UltralyticsDetectionModel(
    model_path="models/yolov8m.pt",     # use valid model
    confidence_threshold=0.12,          # better for aerial humans
    device="cuda"                       # "cpu" if no GPU
)

# -----------------------------
# Read image (NOT VideoCapture)
# -----------------------------
image_path = "image3.jpeg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# -----------------------------
# SAHI sliced prediction
# -----------------------------
result = get_sliced_prediction(
    image,
    model,
    slice_height=512,                   # good balance
    slice_width=512,
    overlap_height_ratio=0.3,
    overlap_width_ratio=0.3,
    postprocess_type="NMS",
    postprocess_match_threshold=0.5,
)

# -----------------------------
# Draw ONLY humans
# -----------------------------
for obj in result.object_prediction_list:

    if obj.category.id != 0:   # person class
        continue

    x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
    score = obj.score.value

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"person {score:.2f}",
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

# -----------------------------
# Display & save output
# -----------------------------
display = cv2.resize(image, None, fx=0.6, fy=0.6)
cv2.imshow("SAHI Image Detection", display)
cv2.imwrite("output_detected.jpg", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
