import cv2
import numpy as np
import config  # For access to the global aircraft_manager


class FindCurrentAircraft:
    """
    Analyzes the bounding boxes of detected aircraft to find the one (the selected aircraft)
    whose average pixel color is closest to the target color.
    """

    def __init__(self, current_aircraft_threshold):
        """
        Initializes the finder with the target color.
        The target color is a bright pink in BGR format.
        """
        self.target_pink_bgr = np.array([103, 94, 230], dtype=np.uint8)
        self.th = current_aircraft_threshold

    def find(self, image):
        """
        Finds the current aircraft on the given image by analyzing the boxes
        obtained from config.aircraft_manager.

        Args:
            image (np.ndarray): The full image in BGR format to be analyzed.

        Returns:
            int or None: The ID of the aircraft with the box that best matches the
                         target color, or None if no aircraft is found.
        """
        # 1. Get all currently tracked boxes from the AircraftManager
        all_boxes = config.aircraft_manager.get_all_boxes()

        if not all_boxes:
            return None

        best_match_id = None
        # We are looking for the minimum color difference, so we initialize it to infinity
        min_avg_color_diff = float('inf')

        # 2. Loop over each aircraft box
        for aircraft_id, bbox in all_boxes.items():
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)

            # Make sure it is within the image boundaries
            h, w, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # If the box is invalid (e.g., outside the screen) or has zero size, skip it
            if x1 >= x2 or y1 >= y2:
                continue

            # 3. Crop the image area within the box (Region of Interest - ROI)
            roi = image[y1:y2, x1:x2]

            # 4. Calculate the average color of the ROI
            # This is the average of the B, G, R values of all pixels in the box.
            average_color_bgr = np.mean(roi, axis=(0, 1))

            # 5. Calculate the difference between the average color and the target color
            # We use the sum of the absolute differences between the color channels (Manhattan distance)
            color_diff = np.sum(np.abs(average_color_bgr - self.target_pink_bgr))

            # 6. If this box is the best match so far, save its ID
            if color_diff < min_avg_color_diff:
                min_avg_color_diff = color_diff
                best_match_id = aircraft_id

        if float(min_avg_color_diff) < self.th:
            return best_match_id

        return None