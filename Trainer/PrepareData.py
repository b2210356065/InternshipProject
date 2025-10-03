import os
import cv2
import numpy as np
import shutil
import yaml
import random
from PIL import Image

# Local project imports
import config
from Trainer.ExtractIcons import ExtractIcons
from Trainer.CreateMapImages import CreateMapImages


class PrepareData:
    """
    Prepares a synthetic dataset for YOLO training by augmenting icons onto map images.
    It handles data generation, structuring, splitting, and saving.
    """

    def __init__(self):
        self.image_generator = CreateMapImages(config.NUM_IMAGES_TO_CREATE)
        self.icon_extractor = ExtractIcons(config.webp_file_path)
        self.num_classes = 0

    def create_yolo_dataset(self):
        """
        Main method to generate and structure the entire YOLO dataset.
        """
        # --- 1. GENERATE SYNTHETIC DATA ---
        synthetic_images, yolo_labels = self._prepare_data()

        # --- 2. SETUP FOLDERS AND CONFIG ---
        dataset_root = 'Trainer/YOLO_Icon_Dataset'
        train_ratio = 0.9

        # Clean up old dataset if it exists
        if os.path.exists(dataset_root):
            shutil.rmtree(dataset_root)

        # Define paths for the YOLOv8 structure
        train_images_path = os.path.join(dataset_root, 'train', 'images')
        train_labels_path = os.path.join(dataset_root, 'train', 'labels')
        val_images_path = os.path.join(dataset_root, 'val', 'images')
        val_labels_path = os.path.join(dataset_root, 'val', 'labels')

        # Create all necessary directories
        for path in [train_images_path, train_labels_path, val_images_path, val_labels_path]:
            os.makedirs(path, exist_ok=True)

        # --- 3. SHUFFLE AND SPLIT THE DATA ---
        indices = list(range(len(synthetic_images)))
        random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]

        # --- 4. SAVE DATA TO FILES ---
        self._save_data_subset(train_indices, synthetic_images, yolo_labels, train_images_path, train_labels_path)
        self._save_data_subset(val_indices, synthetic_images, yolo_labels, val_images_path, val_labels_path)

        # --- 5. CREATE dataset.yaml FILE ---
        class_names = [f'icon_{i}' for i in range(self.num_classes)]
        dataset_yaml_content = {
            'path': os.path.abspath(dataset_root),  # Use absolute path for robustness
            'train': 'train/images',
            'val': 'val/images',
            'nc': self.num_classes,
            'names': class_names
        }
        yaml_path = os.path.join(dataset_root, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml_content, f, sort_keys=False, default_flow_style=False)

        # --- FINAL OUTPUT AS REQUESTED ---
        print(f"Dataset creation complete: {len(train_indices)} training images, {len(val_indices)} validation images.")

    def _save_data_subset(self, indices, all_images, all_labels, images_path, labels_path):
        """Helper function to save a subset of data (train or val) to disk."""
        for i, data_index in enumerate(indices):
            base_filename = f"icon_data_{i}.jpg"

            # Save the image
            image_np = all_images[data_index]
            image_pil = Image.fromarray(image_np)
            image_save_path = os.path.join(images_path, base_filename)
            image_pil.save(image_save_path, 'jpeg', quality=95)

            # Save the corresponding label
            label_str = all_labels[data_index]
            label_filename = os.path.splitext(base_filename)[0] + ".txt"
            label_save_path = os.path.join(labels_path, label_filename)
            with open(label_save_path, 'w') as f:
                f.write(label_str)

    def _prepare_data(self):
        """Generates the augmented images and their corresponding YOLO labels."""
        map_images_list = self.image_generator.get_random_images()
        separated_icons = self.icon_extractor.extract_icons()
        self.num_classes = len(separated_icons)

        synthetic_images = []
        yolo_labels = []

        # Load settings from config
        max_icons_per_image = config.MAX_ICONS_PER_IMAGE
        dataset_multiplier = config.DATASET_MULTIPLIER
        min_icon_scale, max_icon_scale = config.MIN_ICON_SCALE, config.MAX_ICON_SCALE
        blur_levels = config.BLUR_LEVELS

        for _ in range(dataset_multiplier):
            source_background_np = random.choice(map_images_list)
            background_np_uint8 = source_background_np.astype(np.uint8)

            current_map_pil = Image.fromarray(background_np_uint8)
            map_width, map_height = current_map_pil.size
            labels_for_this_image = []
            num_icons_to_place = random.randint(10, max_icons_per_image - 1)

            for _ in range(num_icons_to_place):
                class_id = random.randint(0, self.num_classes - 1)
                base_icon = separated_icons[class_id]

                # --- Apply Augmentations ---
                processed_icon = self._augment_icon(base_icon, blur_levels, min_icon_scale, max_icon_scale)
                if processed_icon is None:
                    continue

                # Paste the augmented icon onto the map
                icon_width, icon_height = processed_icon.size
                if map_width <= icon_width or map_height <= icon_height:
                    continue

                paste_x = random.randint(0, map_width - icon_width)
                paste_y = random.randint(0, map_height - icon_height)
                current_map_pil.paste(processed_icon, (paste_x, paste_y), processed_icon)

                # Calculate YOLO coordinates
                x_center = (paste_x + icon_width / 2) / map_width
                y_center = (paste_y + icon_height / 2) / map_height
                width = icon_width / map_width
                height = icon_height / map_height

                labels_for_this_image.append(f"{class_id} {x_center} {y_center} {width} {height}")

            final_image_rgb = current_map_pil.convert('RGB')
            synthetic_images.append(np.array(final_image_rgb))
            yolo_labels.append("\n".join(labels_for_this_image))

        return synthetic_images, yolo_labels

    def _augment_icon(self, icon_image, blur_levels, min_scale, max_scale):
        """Applies a series of random augmentations to a single icon."""
        # Hue shift
        random_hue = random.randint(0, 179)
        processed_icon = self._change_hue(icon_image, random_hue)

        # Blur
        kernel_size = random.choice(blur_levels)
        if kernel_size > 0:
            rgba_np = np.array(processed_icon)
            rgb = rgba_np[:, :, :3]
            alpha = rgba_np[:, :, 3]
            mask = (alpha == 0).astype(np.uint8) * 255
            inpainted = cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA)
            blurred = cv2.GaussianBlur(inpainted, (kernel_size, kernel_size), 0)
            processed_icon = Image.fromarray(np.dstack((blurred, alpha)), 'RGBA')

        # Scale and Rotation
        scale = random.uniform(min_scale, max_scale)
        angle = random.randint(0, 360)
        new_width = int(processed_icon.width * scale)
        if new_width < 2: return None

        resized = processed_icon.resize((new_width, int(processed_icon.height * scale)), Image.Resampling.LANCZOS)
        rotated = resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        return rotated

    def _change_hue(self, pil_image, target_hue):
        """Changes the hue of a PIL image to a specific target hue value."""
        rgba_image = np.array(pil_image)
        rgb = rgba_image[:, :, :3]
        alpha = rgba_image[:, :, 3]

        # Convert RGB to HSV to manipulate hue
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Change the hue channel (H) for all non-transparent pixels
        hsv[alpha > 0, 0] = target_hue
        # Convert back to RGB
        new_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        new_rgba = np.dstack((new_rgb, alpha))
        return Image.fromarray(new_rgba, 'RGBA')