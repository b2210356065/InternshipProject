from Objects.AircraftManager import AircraftManager
from ImageProcessor import yolov12,  FindCurrentAircraft , OCRProcessor , FrameCreator

# A list of human-readable strings for the different states an object can be in.
conditions = ['Tracking','Lost now','Lost for a while','Object occurs','Not in sight','Current not in sight','Reach the target']
# A dictionary that maps the integer class IDs from the YOLO model to their string names.
cls_names={0: 'icon_0', 1: 'icon_1', 2: 'icon_2', 3: 'icon_3', 4: 'icon_4', 5: 'icon_5', 6: 'icon_6', 7: 'icon_7', 8: 'icon_8'}


# --- General Processing Parameters ---

# The number of frames an object can be lost before it is removed from memory.
memory_time = 30
# Defines the boundaries of the processing area (left, up, right, down). Currently set for a 1920x1080 screen.
l,u,r,d = 0,0,1920,1080
# The number of frames to skip between processing cycles. Used for performance optimization.
skip_frame = 30


# --- File and Model Paths ---

# Path to the input video file that will be processed.
video_path = r'SampleInputOutputs\ekran.mp4'
# Path to the trained YOLOv12 model weights file (.pt).
yolov12_path = r'Models\best.pt'
# Path where the final annotated output video will be saved.
output_video_path = r'SampleInputOutputs\ekran1.mp4'
# Path where the generated PDF summary report will be saved.
pdf_report_path = r'SampleInputOutputs\report.pdf'
# Path to the WebP file containing map icons, likely used for training data generation.
webp_file_path = r'SampleInputOutputs\MapIconsNew.webp'
# Path to save the best model weights during a training session.
BEST_MODEL_SAVE_PATH = r'Models\trained_model.pt'


# --- Algorithm Thresholds ---

# The minimum confidence score for a YOLO detection to be considered a valid object.
add_object_th = 0.5
# The color difference threshold for identifying the currently selected aircraft (lower means stricter color match).
current_aircraft_threshold = 140.0
# The maximum distance in pixels to associate a map text (like an airport code) with a nearby aircraft.
relation_airport_th = 100.0


# --- Static Objects Initialization ---
# These are the core components of the processing pipeline, instantiated once.

# Manages all tracked aircraft, their states, and history.
aircraft_manager = AircraftManager(add_object_th , memory_time , relation_airport_th)
# The YOLOv12 object detector.
yolov12 = yolov12()
# The class responsible for finding the "selected" aircraft based on color.
find_current_aircraft = FindCurrentAircraft(current_aircraft_threshold)
# The class that handles Optical Character Recognition (OCR) for the side panel and map.
ocr_processor = OCRProcessor()
# The class that draws all the annotations (boxes, text, etc.) onto the final frame.
frame_creator = FrameCreator()


#--- Trainer Constants ---
# Parameters specifically for the training data generation script.

# The number of synthetic background images to create for the training dataset.
NUM_IMAGES_TO_CREATE = 100
# The maximum number of icons that will be randomly placed on a single synthetic training image.
MAX_ICONS_PER_IMAGE = 50
# A factor to multiply the dataset size, likely through augmentations.
DATASET_MULTIPLIER = 600
# The minimum and maximum scaling factor for the icons when placing them on training images.
MIN_ICON_SCALE, MAX_ICON_SCALE = 0.2, 0.3
# A list of blur kernel sizes to apply as a data augmentation technique. 0 means no blur.
BLUR_LEVELS = [0, 3, 5]
