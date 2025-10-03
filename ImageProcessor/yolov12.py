import torch

import config
from ultralytics import YOLO


class yolov12:
    def __init__(self):
        """
        Initializes the processor by loading the YOLO model.
        The model object will hold the tracker's state.
        """
        # Load the model only once. The tracker state will be stored within this model object.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLO modeli '{device}' üzerinde çalışacak şekilde yükleniyor.")
        self.model = YOLO(config.yolov12_path).to(device)

    def find_objects(self, image):
        results = self.model.track(
            source=image,
            show=False,  # Don't display the image automatically
            tracker='botsort.yaml',  # Specify your tracker configuration
            conf=0.2,  # Confidence threshold
            iou=0.5,  # IoU threshold for NMS
            persist=True,  # IMPORTANT: This keeps the tracker alive between frames
            verbose=False,  # Set to True for more detailed output
            imgsz=(1088, 1920)  # Specify image size for better performance
        )
        return results[0]
