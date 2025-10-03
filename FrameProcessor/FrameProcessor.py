import config

class FrameProcessor:
    def __init__(self):
        pass

    def process_frame(self, frame):
        # 1. Detect all objects (like aircraft) in the frame using the YOLO model.
        yolo_results = config.yolov12.find_objects(frame)

        # 2. Process the frame with OCR to find the side panel, extract flight data, and read any text on the map.
        ocr_results, panel_boundaries, map_texts = config.ocr_processor.process_image(frame)

        # 3. Update the central aircraft manager with the latest detections, panel info, and map text.
        #    This step handles tracking, updating states (e.g., lost, found), and cleaning up old objects.
        config.aircraft_manager.update(yolo_results, panel_boundaries, map_texts)

        # 4. Identify which of the tracked aircraft is the "currently selected" one (e.g., highlighted with a specific color).
        current_aircraft_id = config.find_current_aircraft.find(frame)

        # 5. Associate the extracted OCR data (flight panel information) with the currently selected aircraft.
        config.aircraft_manager.add_panel_to_aircraft(current_aircraft_id, ocr_results)

        # 6. Create a new frame with all annotations drawn on it (bounding boxes, OCR text, panel lines, etc.).
        new_frame = config.frame_creator.create_annotated_frame(frame, panel_boundaries, map_texts, current_aircraft_id)

        # 7. Return the final, annotated frame for display or saving.
        return new_frame
