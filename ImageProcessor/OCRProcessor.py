import re

import cv2
import numpy as np
import easyocr
from Objects import PanelData


class OCRProcessor:
    """
    Processes an image to extract structured flight data from a side panel.
    It automatically detects the panel boundary, runs OCR on it, and then uses
    positional logic to parse the text into key-value pairs.
    """

    def __init__(self, languages=['en']):
        """
        Initializes the OCRProcessor.
        """
        print("Initializing OCRProcessor: Loading EasyOCR model...")
        self.reader = easyocr.Reader(languages, gpu=True)

        # The threshold value is set to a more reasonable level.
        # This value specifies how "strong" an edge must be to be considered
        # a panel boundary (normalized value).
        self.PANEL_EDGE_STRENGTH_THRESHOLD = 65  # You can test with a value between 60-80.

        print("EasyOCR model loaded successfully.")

    def process_image(self, image):
        # 1. Split the image into a panel and a map. If it fails, it returns panel_boundary_x = None.
        panel_image, map_image, panel_boundary_x = self._split_image_into_panel_and_map(image)

        # 2. Always extract text from the map (whether a panel is found or not).
        map_ocr_results = self._extract_text_from_map(map_image, panel_boundary_x)

        # 3. If the panel was not found (panel_image is None),
        # return with an empty panel object and ONLY the map results.
        if panel_image is None:
            return self._create_panel_data_object({}), None, map_ocr_results

        # 4. If a panel was found, extract and process the text from the panel.
        structured_ocr_results = self._extract_text_from_panel(panel_image)

        if not structured_ocr_results:
            return self._create_panel_data_object({}), panel_boundary_x, map_ocr_results

        flight_data_dict = self._parse_ocr_results(structured_ocr_results)

        # A check in case flight_data_dict is None (to prevent the previous error)
        if flight_data_dict is None:
            flight_data_dict = {}  # Assign an empty dictionary

        panel_data_object = self._create_panel_data_object(flight_data_dict)

        return panel_data_object, panel_boundary_x, map_ocr_results

    def _split_image_into_panel_and_map(self, original_image):
        """
        Uses an "edge strength" threshold for the panel boundary.
        If the strongest edge is below this threshold, it returns None as the panel boundary.
        """
        panel_boundary_x = None  # Initial value is None
        img_h, img_w = original_image.shape[:2]

        try:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            search_x_limit = min(img_w, int(img_w * 0.65))
            vertical_projection = np.sum(edges[:, :search_x_limit], axis=0)

            if vertical_projection.size > 0:
                range_start = int(img_w * 0.15)
                range_end = int(img_w * 0.55)
                if range_end > range_start and range_end <= vertical_projection.size:
                    best_x_candidate = np.argmax(vertical_projection[range_start:range_end]) + range_start
                    peak_strength = vertical_projection[best_x_candidate]
                    normalized_strength = peak_strength / img_h

                    if normalized_strength >= self.PANEL_EDGE_STRENGTH_THRESHOLD:
                        panel_boundary_x = best_x_candidate

        except Exception as e:
            print(f"Error during panel detection: {e}")

        if panel_boundary_x is not None:
            panel_image = original_image[:, 0:panel_boundary_x]
            map_image = original_image[:, panel_boundary_x:]
        else:
            panel_image = None
            map_image = original_image

        return panel_image, map_image, panel_boundary_x

    def _extract_text_from_map(self, map_image, panel_offset_x):
        """
        Handles the case where panel_offset_x is None.
        """
        if map_image is None:
            return []

        # If the panel was not found (panel_offset_x=None), the offset is considered 0.
        offset = panel_offset_x if panel_offset_x is not None else 0

        raw_ocr_output = self.reader.readtext(map_image)
        structured_results = []

        if raw_ocr_output:
            for (box, text, confidence) in raw_ocr_output:
                if confidence < 0.8:
                    continue

                # Adjust coordinates with the 'offset' variable to prevent errors
                global_box_coords = [
                    [int(point[0] + offset), int(point[1])] for point in box
                ]

                x_coords = [p[0] for p in global_box_coords]
                y_coords = [p[1] for p in global_box_coords]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                bbox_xyxy = [x1, y1, x2, y2]

                structured_results.append({
                    "text": text.strip(),
                    "confidence": confidence,
                    "box": bbox_xyxy
                })
        return structured_results

    def _extract_text_from_panel(self, panel_image):
        """
        Performs OCR on the panel image using EasyOCR and formats the output.
        """
        CONFIDENCE_THRESHOLD = 0.4
        raw_ocr_output = self.reader.readtext(panel_image)
        structured_results = []
        if raw_ocr_output:
            for (box, text, confidence) in raw_ocr_output:
                if confidence >= CONFIDENCE_THRESHOLD:
                    box_coords = np.array(box, dtype=np.int32).tolist()
                    structured_results.append({
                        "text": text.strip(),
                        "confidence": confidence,
                        "box": box_coords
                    })
        return structured_results

    def _parse_ocr_results(self, all_ocr_items_ref):
        """
        Parses the structured OCR results using an improved logic.
        This function utilizes text patterns, keywords, and both vertical and
        horizontal positional logic to build a structured dictionary of flight data.
        """

        # --- Helper Functions (Adapted from your existing code) ---
        def get_box_center_y(box):
            return int(np.mean([p[1] for p in box]))

        def get_box_min_x(box):
            return min(p[0] for p in box)

        def get_box_max_x(box):
            return max(p[0] for p in box)

        def get_box_min_y(box):
            return min(p[1] for p in box)

        def get_box_height(box):
            return max(p[1] for p in box) - min(p[1] for p in box)

        # --- NEW and Improved Value-Finding Function ---
        def find_value_for_label(label_item, all_items, processed_indices,
                                 search_below=True, search_right=True,
                                 max_x_distance=150, y_tolerance_ratio=0.8, x_buffer=5):
            """
            Finds the most likely value located below or to the right of a given label.
            """
            label_box = label_item['box']
            label_cy = get_box_center_y(label_box)
            label_cx = (get_box_min_x(label_box) + get_box_max_x(label_box)) / 2
            label_max_x = get_box_max_x(label_box)
            label_max_y = max(p[1] for p in label_box)
            label_height = get_box_height(label_box)

            best_candidate = None
            min_distance = float('inf')

            for idx, item in enumerate(all_items):
                if idx in processed_indices or item is label_item:
                    continue

                item_box = item['box']
                item_cy = get_box_center_y(item_box)
                item_cx = (get_box_min_x(item_box) + get_box_max_x(item_box)) / 2
                item_min_x = get_box_min_x(item_box)
                item_min_y = get_box_min_y(item_box)

                # 1. Search for a value below the label
                if search_below and item_min_y > label_max_y and abs(item_cx - label_cx) < 70:
                    distance = item_min_y - label_max_y
                    if distance < min_distance and distance < 50:  # Vertical distance limit
                        min_distance = distance
                        best_candidate = item

                # 2. Search for a value to the right (only if a better candidate below was not found)
                if search_right and item_min_x > (label_max_x + x_buffer) and abs(item_cy - label_cy) < (
                        label_height * y_tolerance_ratio + 10):
                    distance_x = item_min_x - label_max_x
                    if distance_x < max_x_distance and distance_x < min_distance:
                        min_distance = distance_x
                        best_candidate = item

            return best_candidate

        # --- Main Parsing Logic ---

        # Structure to store the extracted data
        extracted_data = {
            "flight_info": {
                "flight_number": None,
                "airline": None,
                "departure_code": None,
                "arrival_code": None,
                "departure_city": None,
                "arrival_city": None,
            },
            "aircraft_details": {
                "type": None,
                "registration": None,
                "country_of_reg": None,
                "category": None
            },
            "unassigned_texts": []
        }

        processed_items_indices = set()
        all_results_with_indices = list(enumerate(all_ocr_items_ref))

        # 1. Find specific patterns first, like Flight Number, Airline, and Airport Codes, using regex
        airport_codes = []
        for i, item in all_results_with_indices:
            text = item['text']
            # Flight Number (e.g., VY1872)
            if re.match(r'^[A-Z]{2}\d{3,4}$', text) and not extracted_data["flight_info"]["flight_number"]:
                extracted_data["flight_info"]["flight_number"] = text
                processed_items_indices.add(i)
            # Airline
            elif 'vueling' in text.lower():
                extracted_data["flight_info"]["airline"] = "Vueling"
                processed_items_indices.add(i)
            # Airport Codes (3 uppercase letters) within a specific vertical range
            elif re.match(r'^[A-Z]{3}$', text) and 250 < get_box_center_y(item['box']) < 320:
                airport_codes.append(item)
                processed_items_indices.add(i)
            # Airport Names
            elif 'barcelona' in text.lower():
                extracted_data["flight_info"]["departure_city"] = "BARCELONA"
                processed_items_indices.add(i)
            elif 'copenhagen' in text.lower():
                extracted_data["flight_info"]["arrival_city"] = "COPENHAGEN"
                processed_items_indices.add(i)

        # Sort the found airport codes by their x-coordinate (left is departure, right is arrival)
        if len(airport_codes) >= 2:
            airport_codes.sort(key=lambda x: get_box_min_x(x['box']))
            extracted_data["flight_info"]["departure_code"] = airport_codes[0]['text']
            extracted_data["flight_info"]["arrival_code"] = airport_codes[1]['text']

        # 2. Process the remaining general label-value pairs
        labels_map = {
            "AIRCRAFT TYPE": ("aircraft_details", "type"),
            "REGISTRATION": ("aircraft_details", "registration"),
            "COUNTRY OF REG": ("aircraft_details", "country_of_reg"),
            "AIRCRAFT CATEGORY": ("aircraft_details", "category")
        }

        for label_text, (key1, key2) in labels_map.items():
            label_item_info = None
            for i, item in all_results_with_indices:
                if i not in processed_items_indices and label_text.lower() in item['text'].lower():
                    label_item_info = (i, item)
                    break

            if label_item_info:
                label_index, label_item = label_item_info

                # Search for a value both below and to the right of this label
                value_item = find_value_for_label(label_item, all_ocr_items_ref, processed_items_indices)

                if value_item:
                    value_index = all_ocr_items_ref.index(value_item)
                    extracted_data[key1][key2] = value_item['text']

                    processed_items_indices.add(label_index)
                    processed_items_indices.add(value_index)

        # 3. Collect any remaining unassigned texts
        for i, item in all_results_with_indices:
            if i not in processed_items_indices:
                extracted_data["unassigned_texts"].append(item)

        return extracted_data

    def _create_panel_data_object(self, flight_data_dict):
        """
        Takes the parsed data dictionary and creates a PanelData object.
        """
        return PanelData(flight_data_dict)