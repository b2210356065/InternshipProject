import torch
import config
import os
import shutil
from Trainer.PrepareData import PrepareData
from ultralytics import YOLO


class Trainer:
    def __init__(self):
        # This part runs the data preparation when the Trainer is created.
        prepare_data = PrepareData()
        prepare_data.create_yolo_dataset()

    def train(self):
        """
        Trains the YOLO model and saves the best weights to a specified path,
        handling cases where a file with the same name already exists.
        """
        # --- 1. Automatic Device Selection ---
        if torch.cuda.is_available():
            device_name = '0'
            print("CUDA GPU is available. Training will run on GPU device '0'.")
        else:
            device_name = 'cpu'
            print("No CUDA GPU found. Training will run on the CPU.")

        # --- 2. Load the Model and Start Training ---
        y_model = YOLO('Trainer/yolo12s.pt')

        results = y_model.train(
            data='Trainer/YOLO_Icon_Dataset/dataset.yaml',
            epochs=1,
            batch=1,
            imgsz=(1920, 1080),
            scale=0.9,
            mosaic=1.0,
            mixup=0.05,
            copy_paste=0.15,
            dropout=0.3,
            device=device_name,
            rect=True,
            project='Trainer/runs/detect',
            name='training_session'
        )
        print("Training complete.")

        # --- 3. Find and Move the Best Model ---
        temp_run_dir = results.save_dir
        source_model_path = os.path.join(temp_run_dir, 'weights', 'best.pt')

        # Check if the best model was actually created
        if os.path.exists(source_model_path):

            # --- CORRECTED LOGIC FOR UNIQUE FILENAME ---

            # Define the FULL desired path, including the filename.
            # This should ideally come from your config file.
            base_destination_path = config.BEST_MODEL_SAVE_PATH

            # Split this full path into its components
            dir_name = os.path.dirname(base_destination_path)  # -> 'Models'
            base_filename = os.path.basename(base_destination_path)  # -> 'best.pt'
            filename, extension = os.path.splitext(base_filename)  # -> 'best', '.pt'

            # Set the initial destination path to check
            destination_model_path = base_destination_path
            counter = 1

            # Loop ONLY if the destination file already exists
            while os.path.exists(destination_model_path):
                # Create a new filename, e.g., 'best1.pt', 'best2.pt'
                new_filename = f"{filename}{counter}{extension}"
                destination_model_path = os.path.join(dir_name, new_filename)
                counter += 1

            print(f"Moving best model from '{source_model_path}'...")

            # Ensure the destination directory exists (e.g., 'Models/')
            # dir_name will correctly be 'Models' now
            os.makedirs(dir_name, exist_ok=True)

            # Move the file to the unique destination path
            shutil.move(source_model_path, destination_model_path)

            print(f"Model successfully saved to: {destination_model_path}")
            return destination_model_path
        else:
            print(f"Warning: 'best.pt' was not found in {temp_run_dir}. The model could not be saved.")
            return None