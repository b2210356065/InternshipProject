import cv2
import numpy as np
from PIL import Image
from IPython.display import display
import os

class ExtractIcons:
    def __init__(self, webp_file_path):
        self.icon_path = webp_file_path

    def extract_icons(self):
        webp_file_path = self.icon_path
        separated_icons = []
        if not os.path.exists(webp_file_path):
            print(f"ERROR: '{webp_file_path}' cannot be found.")
        else:
            try:
                # 1. Load the image with Pillow (it best handles WEBP and transparency)
                #    Ensure it's 4-channel (Red, Green, Blue, Alpha) with .convert('RGBA')
                pil_image = Image.open(webp_file_path).convert('RGBA')

                # 2. Convert the image to a NumPy array for OpenCV to process
                #    OpenCV uses the BGRA format instead of RGBA, so convert the channels
                opencv_image_rgba = np.array(pil_image)
                opencv_image_bgra = cv2.cvtColor(opencv_image_rgba, cv2.COLOR_RGBA2BGRA)

                # 3. Use only the Alpha (transparency) channel to find the objects
                #    In BGRA, the alpha channel is the 4th channel (index 3)
                alpha_channel = opencv_image_bgra[:, :, 3]

                # 4. Convert the alpha channel to a binary image (black/white)
                #    Every pixel with Alpha > 0 (i.e., not transparent) will become white (255)
                _, thresh = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

                # 5. Find the external contours (object boundaries) in the binary image
                #    cv2.RETR_EXTERNAL finds only the outermost boundaries, which is exactly what we want.
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                print(f"Contour analysis complete. {len(contours)} potential objects found.")
                print("Separating icons...")

                # 6. Process each found contour (aircraft)
                for contour in contours:
                    # Draw a bounding box around the contour
                    x, y, w, h = cv2.boundingRect(contour)

                    # Check to filter out very small noise
                    if w > 10 and h > 10:  # Take those with a width and height greater than 10 pixels
                        # Crop this box from the ORIGINAL Pillow image
                        # Pillow's .crop() method preserves transparency
                        # Crop format: (left, top, right, bottom)
                        cropped_icon_pil = pil_image.crop((x, y, x + w, y + h))

                        # Add the cropped transparent icon to the list
                        separated_icons.append(cropped_icon_pil)

                # --- SHOWING THE RESULT ---
                if separated_icons:
                    print(f"\nSuccessfully separated and added {len(separated_icons)} icons to the list.")
                    print("Here are the separated icons:")
                    return separated_icons
                else:
                    print("No icons could be separated. Check the file's content.")
                    return None

            except Exception as e:
                print(f"An error occurred: {e}")
                return None