import config
import math


class Aircraft():
    def __init__(self, id, bbox, conf ,cls_id):
        self._id = id
        self._bbox = bbox
        self._conf = conf
        self._cls_id = cls_id

        self._lost_time = 0
        self._condition = 0
        self._location = 'Unknown'
        self._panel = None
        self._past_bbox = None
        self._velocity = None
        self._direction = None
        self._max_conf = conf

    # Getter and Setter Functions
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def bbox(self):
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value
    @property
    def conf(self):
        return self._conf
    @conf.setter
    def conf(self, value):
        self._conf = value
    @property
    def cls_id(self):
        return self._cls_id

    @cls_id.setter
    def cls_id(self, value):
        self._cls_id = value
    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        self._location = value
    @property
    def panel(self):
        return self._panel

    @panel.setter
    def panel(self, value):
        self._panel = value

    @property
    def lost_time(self):
        return self._lost_time

    @lost_time.setter
    def lost_time(self, value):
        self._lost_time = value

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, value):
        self._condition = value

    @property
    def past_bbox(self):
        return self._past_bbox

    @past_bbox.setter
    def past_bbox(self, value):
        self._past_bbox = value
    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        self._direction = value

    def update(self, bbox, conf, cls_id):
        if conf == 0:  # This block handles the case where the object is NOT detected in the current frame (it is "lost").
            if self.lost_time == 0:  # If this is the very first frame the object is lost.
                self.condition = 1  # Set condition to 1 ("Just Lost").
                if not self.is_in_sight():  # Check if the object was lost near the edge of the screen.
                    self.condition = 5  # If so, set condition to 5 ("Lost Out of Sight").
            else:  # If the object was already lost in previous frames.
                self.condition = 2  # Set condition to 2 ("Continuously Lost").
                if self.condition == 5:  # Check if the object's last known state was "Lost Out of Sight".
                    self.condition = 4  # Update condition to 4 (perhaps "Confirmed Out of Sight").
            self.lost_time += 1  # Increment the counter for consecutive lost frames.
        else:  # This block handles the case where the object IS detected in the current frame.
            if self.lost_time != 0:  # If the object was previously lost but is now found again.
                self.lost_time = 0  # Reset the lost frame counter.
                self.condition = 3  # Set condition to 3 ("Re-detected").
            else:  # If the object was tracked in the previous frame and is still being tracked.
                self.condition = 0  # Set condition to 0 ("Normal Tracking").

        if self.panel is not None:  # If the aircraft has associated panel data from OCR.
            # Calculate velocity and direction based on bounding box history.
            self.velocity, self.direction = self.find_px_velocity()

        self.past_bbox = self.bbox  # Store the current bounding box as the 'past' for the next update.
        self.bbox = bbox  # Update the current bounding box with the new data.
        # Recalculate direction based on the updated position.
        _, self.direction = self.find_px_velocity()

        self._conf = conf  # Update the current confidence score.
        if conf >= self._max_conf:  # If the new confidence is the highest we've seen for this object.
            self._max_conf = conf  # Update the maximum confidence score.
            self.cls_id = cls_id  # Update the class ID, assuming the highest confidence detection is the most accurate.

    def find_px_velocity(self):
        if self.past_bbox is None or self.bbox is None:
            return None, None

        # Difference between the average center points (dx, dy)
        dx = (self.bbox[0] + self.bbox[2]) // 2 - (self.past_bbox[0] + self.past_bbox[2]) // 2
        dy = (self.bbox[1] + self.bbox[3]) // 2 - (self.past_bbox[1] + self.past_bbox[3]) // 2

        # Total velocity (magnitude of the vector)
        total_velocity = math.sqrt(dx ** 2 + dy ** 2)

        # Angle calculation (up direction is 0 degrees, counter-clockwise)
        # math.atan2(-dy, dx) is used because in screen coordinates, the y-axis points downwards.
        # Using -dy converts it to a mathematical coordinate system (y-axis points upwards).
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize the angle to be in the range [0, 360)
        direction_angle = (angle_deg + 360) % 360

        return total_velocity, direction_angle

    def is_in_sight(self):
        if self.past_bbox is None:
            return True

        x1, y1, x2, y2 = self.past_bbox[0], self.past_bbox[1], self.past_bbox[2], self.past_bbox[3]

        # Get angle values from the find_px_velocity function
        _, angle = self.find_px_velocity()

        # Tendency to go left (between 225 and 315 degrees)
        if x1 <= config.l and 225 < angle < 315:
            return False
        # Tendency to go right (between 45 and 135 degrees)
        elif x2 >= config.r and 45 < angle < 135:
            return False
        # Tendency to go up (between 315 and 45 degrees)
        elif y1 <= config.u and (angle > 315 or angle < 45):
            return False
        # Tendency to go down (between 135 and 225 degrees)
        elif y2 >= config.d and 135 < angle < 225:
            return False

        return True