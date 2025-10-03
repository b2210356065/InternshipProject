import cv2
import numpy as np
import random
import config  # For access to the global aircraft_manager


class FrameCreator:
    def __init__(self):
        # Colors for class IDs (in BGR format)
        self.class_colors = {
            0: (255, 0, 0),  # Blue
            1: (0, 0, 255),  # Red
            2: (0, 255, 255),  # Yellow
            3: (255, 0, 255),  # Magenta
            4: (0, 255, 0),  # Green
            5: (255, 255, 0),  # Cyan
            6: (0, 165, 255),  # Orange
            7: (128, 0, 128),  # Purple
            8: (255, 255, 255),  # White
        }
        # Names for class IDs (update according to your YOLO model)
        self.class_names = config.cls_names
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.box_thickness = 2

    def _get_color_for_class(self, cls_id):
        """
        Gets a specific color for a class ID. If the class is new,
        it generates and stores a random color for it.
        """
        color = self.class_colors.get(cls_id)
        if color is None:
            # Create and save a random color for an undefined class
            color = [random.randint(0, 255) for _ in range(3)]
            self.class_colors[cls_id] = color
        return color

    def create_annotated_frame(self, frame, panel_boundary_x, map_texts, selected_aircraft_id=None):
        """
        Creates the final display frame by drawing all annotations.
        """
        annotated_frame = frame.copy()
        self._draw_all_object_boxes(annotated_frame)
        self._draw_map_texts(annotated_frame, map_texts)

        if selected_aircraft_id is not None:
            self._highlight_selected_aircraft(annotated_frame, selected_aircraft_id)

        self._draw_panel_box(annotated_frame, panel_boundary_x)

        return annotated_frame

    def _draw_all_object_boxes(self, image):
        """
        Draws bounding boxes and labels for all tracked aircraft.
        """
        all_aircrafts = config.aircraft_manager.get_all_aircrafts()
        if not all_aircrafts:
            return

        for aircraft in all_aircrafts:
            if aircraft.bbox is None:
                continue

            x1, y1, x2, y2 = map(int, aircraft.bbox)
            cls_id = int(aircraft.cls_id)
            track_id = aircraft.id
            conf = aircraft.conf

            # Get the class-specific color
            color = self._get_color_for_class(cls_id)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.box_thickness)

            # Create the label text
            class_name = self.class_names.get(cls_id, f'Class {cls_id}')
            label = f"ID:{track_id} {class_name} {conf:.2f}"

            # Draw a background box for the text
            (w, h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            # Write the text
            cv2.putText(image, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

    def _highlight_selected_aircraft(self, image, selected_id):
        """
        Draws a special, thicker box around the currently selected aircraft.
        """
        selected_aircraft = config.aircraft_manager.get_aircraft(selected_id)
        if selected_aircraft is None or selected_aircraft.bbox is None:
            return

        x1, y1, x2, y2 = map(int, selected_aircraft.bbox)

        # Use a bright green color and a thicker frame for highlighting
        highlight_color = (0, 255, 0)
        highlight_thickness = 4

        cv2.rectangle(image, (x1, y1), (x2, y2), highlight_color, highlight_thickness)
        cv2.putText(image, "SELECTED", (x1, y2 + 20), self.font, 0.7, highlight_color, 2)

    def _draw_panel_box(self, image, panel_boundary_x):
        """
        Draws a box around the detected OCR panel area.
        """
        if panel_boundary_x is None:
            return
        h = image.shape[0]
        panel_color = (255, 255, 0)  # Cyan
        panel_thickness = 3

        cv2.rectangle(image, (0, 0), (panel_boundary_x, h - 1), panel_color, panel_thickness)
        cv2.putText(image, "PANEL", (10, 30), self.font, 1, panel_color, 2)

    def _draw_map_texts(self, image, map_texts):
        """
        Draws the text detected on the map area of the screen with a pale color.
        """
        if not map_texts:
            return

        # A pale and non-prominent color (light gray in BGR format)
        pale_color = (180, 180, 180)
        map_font_scale = 0.5
        map_font_thickness = 1

        for text_info in map_texts:
            text = text_info['text']
            box = text_info['box']
            x1, y1, x2, y2 = box

            # Print the text at the top-left corner of the box
            cv2.putText(image, text, (x1, y1 - 5), self.font, map_font_scale, pale_color, map_font_thickness)