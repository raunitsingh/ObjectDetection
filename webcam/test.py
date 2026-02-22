from ultralytics import YOLO
import cv2
import numpy as np

# ==========================
# CONFIGURATION
# ==========================
MODEL_PATH = "yolov8s.pt"         # <-- your trained aerial model
VIDEO_PATH = "yt7.mp4"
IMG_SIZE = 1280                 # IMPORTANT for small objects
CONF_THRESHOLD = 0.05            # Low confidence for aerial view
TARGET_CLASS = 0                # person
TILE_SIZE = 640                 # tile size for frame splitting
OVERLAP = 0.2                   # tile overlap

# ==========================
# LOAD MODEL
# ==========================
model = YOLO("yolov8s.pt")

# ==========================
# VIDEO CAPTURE
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Could not open video")

print("✅ Video loaded successfully")

# ==========================
# FRAME TILING FUNCTION
# ==========================
def tile_frame(frame, tile_size=640, overlap=0.2):
    h, w, _ = frame.shape
    step = int(tile_size * (1 - overlap))
    tiles = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = frame[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] < 200 or tile.shape[1] < 200:
                continue
            tiles.append((tile, x, y))

    return tiles

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Video ended")
        break

    output_frame = frame.copy()

    # ---- TILE THE FRAME ----
    tiles = tile_frame(frame, TILE_SIZE, OVERLAP)

    for tile, offset_x, offset_y in tiles:

        # ---- YOLO DETECTION ----
        results = model(
            tile,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=0,        # change to "cpu" if no GPU
            verbose=False
        )

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls != TARGET_CLASS:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ---- MAP TILE COORDS TO ORIGINAL FRAME ----
                x1 += offset_x
                x2 += offset_x
                y1 += offset_y
                y2 += offset_y

                # ---- DRAW BOUNDING BOX ----
                cv2.rectangle(
                    output_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output_frame,
                    f"Person {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # ---- DISPLAY ----
    cv2.imshow("Aerial Person Detection", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
