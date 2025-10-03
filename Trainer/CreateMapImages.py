import random
import numpy as np
from staticmap import StaticMap
from PIL import Image
import os


class CreateMapImages:
    """
    Generates and holds a list of random map images as NumPy arrays.
    The images are created using the staticmap library and are in standard RGB format.
    """

    def __init__(self, num_images_to_create):
        # Configuration parameters
        self.num_images_to_create = num_images_to_create
        self.image_width = 1920
        self.image_height = 1080
        self.min_zoom = 7
        self.max_zoom = 15
        self.save_directory = 'MapImages/'

        # Geographic constants for generating coordinates
        self.TURKEY_REGION_BOX = {
            "min_lat": 36.0, "max_lat": 42.0,
            "min_lon": 26.0, "max_lon": 45.0
        }
        self.WORLD_CITIES = [
            (51.50, -0.12), (40.71, -74.00), (35.68, 139.69),
            (-33.86, 151.20), (48.85, 2.35), (34.05, -118.24)
        ]

        # This list will hold all the generated map images as NumPy arrays
        self.map_images = []

        # Generate the images immediately when the object is created
        self._generate_images()

        # Final, clean print statement as requested
        print(f"{len(self.map_images)} map images were created successfully.")

    def _get_land_focused_coords(self):
        """Generates random geographic coordinates, biased towards land masses."""
        # 80% chance to generate coordinates within the Turkey region box
        if random.random() < 0.80:
            lat = random.uniform(self.TURKEY_REGION_BOX["min_lat"], self.TURKEY_REGION_BOX["max_lat"])
            lon = random.uniform(self.TURKEY_REGION_BOX["min_lon"], self.TURKEY_REGION_BOX["max_lon"])
        # 20% chance to pick a major world city and randomize slightly around it
        else:
            base_lat, base_lon = random.choice(self.WORLD_CITIES)
            lat = base_lat + random.uniform(-0.5, 0.5)
            lon = base_lon + random.uniform(-0.5, 0.5)
        return lat, lon

    def _generate_images(self):
        """
        Internal method that loops and creates the map images,
        storing them in the self.map_images list.
        """
        os.makedirs(self.save_directory, exist_ok=True)

        for i in range(self.num_images_to_create):
            try:
                # Create a map object with specified dimensions
                m = StaticMap(self.image_width, self.image_height)

                # Get random coordinates and zoom level
                lat, lon = self._get_land_focused_coords()
                zoom = random.randint(self.min_zoom, self.max_zoom)

                # Render the map. This returns a Pillow (PIL) Image object in RGB format.
                pil_image = m.render(zoom=zoom, center=(lon, lat))
                np_image = np.array(pil_image.convert('RGB'), dtype=np.uint8)

                # Add the correctly formatted NumPy array to our list
                self.map_images.append(np_image)

                # Optional: Save the image file to disk
                file_name = f"map_{i + 1}_zoom{zoom}.png"
                full_path = os.path.join(self.save_directory, file_name)
                pil_image.save(full_path)

            except Exception as e:
                # This is the only print inside the loop, for essential error feedback
                print(f"Error: Image {i + 1} could not be created and was skipped. Reason: {e}")

    def get_random_images(self):
        """
        Public method to access the list of all generated map images.
        This is the function that PrepareData should call.

        Returns:
            list: A list containing all generated map images as NumPy arrays.
        """
        return self.map_images