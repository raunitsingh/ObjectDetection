import random
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel


class SAHIObjectDetection:
    def __init__(
        self,
        model_path,
        confidence=0.25,
        device="cpu",
        slice_height=512,
        slice_width=512,
        overlap=0.2,
    ):

        self.model = UltralyticsDetectionModel(
            model_path=model_path,
            confidence_threshold=confidence,
            device=device,
        )

        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap = overlap

        # random colors for up to 80 classes
        self.colors = {
            i: tuple(random.randint(0, 255) for _ in range(3))
            for i in range(80)
        }

    def detect(self, frame):
        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
        )
        return result.object_prediction_list
