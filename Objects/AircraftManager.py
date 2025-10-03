import math

from Objects import Aircraft , PanelData

class AircraftManager:
    def __init__(self,add_object_th , memory_time , relation_airport_th):
        self.aircrafts = {} # A dictionary to store Aircraft objects, with their ID as the key.
        self.th = add_object_th # Confidence threshold for adding a new object.
        self.memory_time = memory_time # How long to keep a lost aircraft in memory before deleting.
        self.relation_airport_th = relation_airport_th # Distance threshold to associate an aircraft with an airport text.
    def add_or_update_aircraft(self, aircraft_id, bbox, conf , cls_id):
        if aircraft_id in self.aircrafts:
            # If the aircraft already exists, call its update method with the new data.
            self.aircrafts[aircraft_id].update(bbox,conf ,cls_id)
        else:
            # If it's a new ID, create a new Aircraft object and add it to the dictionary.
            new_aircraft = Aircraft(id=aircraft_id, bbox=bbox, conf = conf,cls_id=cls_id)
            self.aircrafts[aircraft_id] = new_aircraft

    def get_aircraft(self, aircraft_id):
        # Retrieve a single aircraft object by its ID.
        return self.aircrafts.get(aircraft_id)

    def remove_aircraft(self, aircraft_id):
        # Delete an aircraft from the manager.
        if aircraft_id in self.aircrafts:
            del self.aircrafts[aircraft_id]
            return True # Return True on successful removal.
        else:
            return False # Return False if the aircraft was not found.

    def get_all_aircrafts(self):
        # Return a list of all current Aircraft objects.
        return list(self.aircrafts.values())
    def get_all_boxes(self):
        # Return a dictionary of all current aircraft IDs and their bounding boxes.
        all_boxes = {}
        for id,aircraft in self.aircrafts.items():
            all_boxes[id] = aircraft.bbox
        return all_boxes



    def update(self, result , panel_boundaries , map_texts):
        track_id_list = [] # Keep track of all object IDs seen in the current frame.
        if result.boxes.id is None:
            return # Exit if there are no tracked objects in the result.
        for i in range(len(result.boxes.id)):
            bbox = result.boxes.xyxy[i].tolist() # Get the bounding box coordinates.
            if self._is_in(bbox, panel_boundaries): # Check if the object is within the side panel area.
                continue # If it's a panel element, ignore it.
            conf = result.boxes.conf[i].item() # Get the detection confidence.
            cls = result.boxes.cls[i].item() # Get the class ID.
            track_id = result.boxes.id[i].int().item() # Get the unique tracking ID.
            track_id_list.append(track_id) # Add the ID to the list for this frame.
            if conf > self.th:
                # If confidence is high enough, add or update the aircraft.
                self.add_or_update_aircraft(track_id, bbox, conf, cls)

        # Find which aircraft were being tracked but were not detected in this frame.
        lost_ids = self._find_lost_aircraft(track_id_list)

        for track_id in lost_ids:
            # For each lost aircraft, update its status (bbox=None, conf=0).
            self.add_or_update_aircraft(track_id, None, 0, None)
        self._cleanup_lost_aircrafts() # Clean up aircraft that have been lost for too long.
        self._determine_aircrafts_locations(map_texts) # Try to associate aircraft with nearby text on the map.

    def add_panel_to_aircraft(self, aircraft_id, panel_data):
        # Assigns PanelData (from OCR) to a specific aircraft.
        if aircraft_id is None or panel_data is None:
            return False # Fail if inputs are invalid.
        aircraft = self.get_aircraft(aircraft_id)
        aircraft.panel = panel_data

    def _is_in(self , box1 , border):
        # A simple check to see if a box is completely to the left of a vertical border line.
        if border is None or box1 is None:
            return False # Can't check if border or box is missing.
        if (box1[2] < border): # box1[2] is the x2 coordinate (right side of the box).
            return True
        return False
    def _find_lost_aircraft(self, current_track_ids):
        # Compares the set of all managed aircraft with the set of aircraft seen in the current frame.
        managed_ids = set(self.aircrafts.keys())
        current_ids = set(current_track_ids)

        lost_ids = managed_ids - current_ids # The difference is the set of lost aircraft IDs.

        return lost_ids

    def _get_box_center(self,box):
        # Calculates the center point of a bounding box.
        if box is None:
            return None
        return [box[0]+box[2] , box[1]+box[3]]

    def _distance(self , center1 , center2):
        # Calculates the Euclidean distance between two center points.
        if center1 is None or center2 is None:
            return math.inf # Return infinity if one of the points is invalid.
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance

    def _determine_aircrafts_locations(self, map_texts):
        # Creates a set of aircraft that are currently in a "lost" state.
        currently_lost_aircrafts = set()
        for id, aircraft in self.aircrafts.items():
            if aircraft.condition == 1: # Condition 1 typically means 'lost'.
                currently_lost_aircrafts.add(id)
        # Iterate through all texts found on the map.
        for text in map_texts:
            # Iterate through all managed aircraft.
            for id, aircraft in self.aircrafts.items():
                c1 = self._get_box_center(aircraft.bbox) # Aircraft's center.
                c2 = self._get_box_center(text['box']) # Text's center.

                centers_distance = self._distance(c1, c2) # Calculate distance between them.

                if centers_distance < self.relation_airport_th:
                    # If the text is close enough to the aircraft, assume it's its location.
                    aircraft.location = text['text']
                    if id in currently_lost_aircrafts:
                        # If a lost aircraft is now near a location, update its status.
                        aircraft.condition = 6 # Condition 6 might mean 're-identified at location'.
    def _cleanup_lost_aircrafts(self):
        # It's important to create a separate list of IDs to remove,
        # because you can't modify a dictionary while iterating over it.
        lost_ids = [
            aircraft_id for aircraft_id, aircraft in self.aircrafts.items()
            if aircraft.lost_time > self.memory_time # Check if the lost time exceeds the memory limit.
        ]

        for aircraft_id in lost_ids:
            # Remove each aircraft that has been lost for too long.
            self.remove_aircraft(aircraft_id)