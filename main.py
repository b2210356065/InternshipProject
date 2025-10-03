# main.py

import streamlit as st
from tqdm import tqdm

from Interface.Interface import Interface
# Local project imports
from Video import VideoProcessor
from FrameProcessor import FrameProcessor
from Report import Report
import config
from Trainer.Trainer import Trainer
import os

class Main:
    """
    Main application class that orchestrates the entire video processing workflow.
    """

    def __init__(self):
        self.video_processor = VideoProcessor()
        self.frame_processor = FrameProcessor()
        self.report_generator = Report()

    def run_video_processing(self):
        """
        Executes the main video processing pipeline and returns the raw data of the outputs.
        """
        print("--- Step 1: Extracting video frames ---")
        self.video_processor.extract_frames()
        extracted_frames = self.video_processor.get_frames()
        if not extracted_frames:
            print("Error: No frames were extracted.")
            return None, None

        print(f"Successfully extracted {len(extracted_frames)} frames.")
        print("\n--- Step 2: Processing each frame ---")
        processed_frames_list = []
        for frame in tqdm(extracted_frames, desc="Processing Frames"):
            processed_frame = self.frame_processor.process_frame(frame)
            processed_frames_list.append(processed_frame)
            self.report_generator.log_frame_data()

        print("\n--- Step 3: Assembling the output video ---")
        self.video_processor.create_new_video(
            config.output_video_path,
            frames_to_use=processed_frames_list
        )
        print(f"Output video saved to: {config.output_video_path}")
        print("\n--- Step 4: Generating final reports ---")
        self.report_generator.generate_pdf_report()
        print("\n>>> All video processing operations completed successfully. <<<")

        video_bytes = None
        pdf_bytes = None

        if os.path.exists(config.output_video_path):
            with open(config.output_video_path, 'rb') as f:
                video_bytes = f.read()

        if os.path.exists(config.pdf_report_path):
            with open(config.pdf_report_path, 'rb') as f:
                pdf_bytes = f.read()

        return video_bytes, pdf_bytes


# --- Main execution block ---
if __name__ == "__main__":

    app = Interface()
    start_training = app.run()

    if start_training:
        settings = st.session_state.get('train_settings', {})
        app.update_config_file(settings)

        print("Starting the training process...")
        trainer_app = Trainer()
        trainer_app.train()
        st.session_state.status_message = "Training completed successfully!"


        st.session_state.is_busy = False
        st.rerun()

    elif st.session_state.get('is_busy') and st.session_state.get('run_settings'):
        settings = st.session_state.get('run_settings', {})
        app.update_config_file(settings)

        print("Starting the video processing pipeline...")
        main_app = Main()
        # --- DEĞİŞİKLİK: Artık dosya yolları değil, byte'lar alınıyor ---
        video_bytes, pdf_bytes = main_app.run_video_processing()

        st.session_state.status_message = "Video processing completed successfully!"

        # --- DEĞİŞİKLİK: Byte'ları session_state'e kaydet ---
        st.session_state.last_video_bytes = video_bytes
        st.session_state.last_pdf_bytes = pdf_bytes
        st.session_state.run_completed = True



        st.session_state.is_busy = False
        st.session_state.run_settings = None
        st.rerun()