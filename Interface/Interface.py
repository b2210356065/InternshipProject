# interface.py

import streamlit as st
import os
import time
import shutil
import re
import config
import base64


class Interface:
    def __init__(self):
        if 'page' not in st.session_state:
            st.session_state.page = 'main'
        if 'is_busy' not in st.session_state:
            st.session_state.is_busy = False
        if 'status_message' not in st.session_state:
            st.session_state.status_message = ""
        # --- DEĞİŞİKLİK: Path yerine bytes saklayacağız ---
        if 'last_video_bytes' not in st.session_state:
            st.session_state.last_video_bytes = None
        if 'last_pdf_bytes' not in st.session_state:
            st.session_state.last_pdf_bytes = None
        # ---
        if 'run_completed' not in st.session_state:
            st.session_state.run_completed = False
        # Model yönetimi için state'ler
        if 'model_to_rename' not in st.session_state:
            st.session_state.model_to_rename = None
        if 'new_model_name' not in st.session_state:
            st.session_state.new_model_name = ""
        if 'model_to_delete' not in st.session_state:
            st.session_state.model_to_delete = None

    def run(self):
        """
        Streamlit arayüzünü çalıştırır ve sayfa durumuna göre ilgili arayüzü çizer.
        'Train' butonuna basıldığında True, diğer durumlarda False döndürür.
        """
        if st.session_state.page == 'train':
            return self._draw_train_page()
        else:  # Main page
            self._draw_main_page()
            return False

    # --- Main Page UI ---
    def _draw_main_page(self):
        st.title("Icon Detection Video Processor")

        # Eğer bir işlem devam ediyorsa, sadece işlem ekranını göster
        if st.session_state.is_busy:
            self._draw_processing_ui()
        else:
            # Eğer bir işlem YENİ BİTTİYSE, en üste başarı mesajını bas
            if st.session_state.status_message:
                st.success(st.session_state.status_message)
                st.session_state.status_message = ""  # Mesajı gösterdikten sonra temizle

            # Ana ayarlar arayüzünü çiz
            self._draw_settings_ui()

            # Eğer bir "run" işlemi tamamlandıysa, sonuçları göster
            if st.session_state.run_completed:
                self._display_previous_results()

    # --- Settings UI (Main Page) ---
    def _draw_settings_ui(self):
        st.sidebar.button("Train Your Model", on_click=self._navigate_to_train, use_container_width=True,
                          disabled=st.session_state.is_busy)

        st.header("1. Path Selection")
        video_path = st.text_input("Source Video Path", value=config.video_path)
        output_video_path = st.text_input("Output Video Path", value=config.output_video_path)
        pdf_report_path = st.text_input("PDF Report Path", value=config.pdf_report_path)

        st.subheader("YOLO Model Selection")
        selected_model = self._draw_model_manager()

        st.header("2. Processing Parameters")
        col1, col2 = st.columns(2)
        memory_time = col1.number_input("Memory Time (seconds)", min_value=1, value=config.memory_time, step=1)
        skip_frame = col2.number_input("Frame Skip Interval", min_value=1, value=config.skip_frame, step=1)

        st.header("3. Start Process")
        if st.button("Start Processing", use_container_width=True, type="primary", disabled=st.session_state.is_busy):
            if selected_model:
                st.session_state.run_settings = {
                    "video_path": video_path, "yolov12_path": selected_model, "output_video_path": output_video_path,
                    "pdf_report_path": pdf_report_path, "memory_time": int(memory_time), "skip_frame": int(skip_frame)
                }
                st.session_state.is_busy = True
                st.session_state.status_message = "Processing video..."
                st.session_state.run_completed = False  # Yeni işlem başlarken eski sonuçları gizle
                st.rerun()

    # --- Processing UI ---
    def _draw_processing_ui(self):
        st.header(st.session_state.status_message)
        st.info("The application is currently busy. Please wait until the process is complete.")
        st.progress(0)  # Bu bar, dışarıdan (tqdm hook ile) güncellenebilir
        st.code("Processing logs will appear in the terminal...", language="text")

    # --- Train Page UI ---
    def _draw_train_page(self):
        st.title("Train Your Model")

        # Eğer bir işlem devam ediyorsa veya yeni bittiyse (ve ana sayfaya yönlendirildiyse)
        if st.session_state.is_busy:
            self._draw_processing_ui()
            return False

        st.header("1. Dataset Parameters")
        webp_file_path = st.text_input("Icon Source Path (.webp)", value=config.webp_file_path)
        num_images = st.number_input("Number of Maps to Generate", min_value=1, value=config.NUM_IMAGES_TO_CREATE,
                                     step=1)
        dataset_multiplier = st.number_input("Dataset Multiplier", min_value=1, value=config.DATASET_MULTIPLIER, step=1)

        st.header("2. Icon Augmentation")
        max_icons = st.number_input("Max Icons per Image", min_value=1, value=config.MAX_ICONS_PER_IMAGE, step=1)
        col1, col2 = st.columns(2)
        min_scale = col1.number_input("Min Icon Scale", min_value=0.1, value=config.MIN_ICON_SCALE, step=0.05,
                                      format="%.2f")
        max_scale = col2.number_input("Max Icon Scale", min_value=0.1, value=config.MAX_ICON_SCALE, step=0.05,
                                      format="%.2f")

        st.header("3. Model Output")
        model_name = st.text_input("New Model Name (without .pt extension)", value="trained_model")

        if st.button("Start Training", use_container_width=True, type="primary", disabled=st.session_state.is_busy):
            st.session_state.train_settings = {
                "webp_file_path": webp_file_path, "NUM_IMAGES_TO_CREATE": int(num_images),
                "DATASET_MULTIPLIER": int(dataset_multiplier), "MAX_ICONS_PER_IMAGE": int(max_icons),
                "MIN_ICON_SCALE": float(min_scale), "MAX_ICON_SCALE": float(max_scale),
                "model_name": model_name
            }
            st.session_state.is_busy = True
            st.session_state.status_message = "Training model..."
            st.session_state.page = 'main'  # İşlem ekranını göstermek için ana sayfaya yönlendir
            return True  # main.py'e eğitimi başlatma sinyali gönder

        if st.button("← Back to Main Page"):
            self._navigate_to_main()

        return False

    # --- Helper Functions ---
    def _navigate_to_train(self):
        st.session_state.page = 'train'
        st.rerun()

    def _navigate_to_main(self):
        st.session_state.page = 'main'
        st.rerun()

    def update_config_file(self, settings_dict):
        if 'model_name' in settings_dict:
            model_name = settings_dict.pop('model_name')
            if model_name:
                full_model_path = os.path.join('Models', model_name + '.pt')
                settings_dict['BEST_MODEL_SAVE_PATH'] = full_model_path

        config_path = "config.py"
        with open(config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            updated = False
            for key, value in settings_dict.items():
                if re.match(fr"^\s*{key}\s*=", line):
                    if isinstance(value, str):
                        new_lines.append(f"{key} = r'{os.path.abspath(value)}'\n")
                    elif isinstance(value, tuple):
                        new_lines.append(f"{key} = {value[0]}, {value[1]}\n")
                    else:
                        new_lines.append(f"{key} = {value}\n")
                    updated = True
                    break
            if not updated:
                new_lines.append(line)

        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        st.toast("Configuration updated!")

    def _display_previous_results(self):
        """
        Displays the last generated video and PDF report directly from byte data
        stored in the session state.
        """
        st.markdown("---")
        st.header("Last Run Results")

        # --- VİDEOYU GÖRÜNTÜLE ---
        video_bytes = st.session_state.get('last_video_bytes')
        if video_bytes:
            st.video(video_bytes)
            st.info(f"Showing last processed video.")
        else:
            st.warning("No video data from the last run to display.")

        # --- PDF'İ GÖRÜNTÜLE ---
        pdf_bytes = st.session_state.get('last_pdf_bytes')
        if pdf_bytes:
            try:
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

                st.subheader("PDF Report")
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.info(f"Showing last generated report.")
            except Exception as e:
                st.error(f"Could not display PDF report from data: {e}")
        else:
            st.warning("No PDF data from the last run to display.")

    def _draw_model_manager(self):
        models_dir = "Models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        try:
            models = sorted([f for f in os.listdir(models_dir) if f.endswith('.pt')])
            model_names_no_ext = [os.path.splitext(m)[0] for m in models]
        except Exception as e:
            st.error(f"Could not read models from '{models_dir}': {e}")
            return None

        if not models:
            st.info("No models (.pt files) found in 'Models' directory. Train a model to begin.")
            return None

        selected_model_name = st.selectbox("Select a model from the list below", options=model_names_no_ext)

        with st.expander("Model Management"):
            for model_file in models:
                model_name_no_ext = os.path.splitext(model_file)[0]
                col_name, col_rename, col_delete = st.columns([4, 1, 1])
                col_name.write(f"- `{model_name_no_ext}`")

                if col_rename.button("Rename", key=f"rename_{model_file}", use_container_width=True):
                    st.session_state.model_to_rename = model_file
                    st.session_state.new_model_name = model_name_no_ext
                    st.rerun()

                if col_delete.button("Delete", key=f"delete_{model_file}", use_container_width=True):
                    st.session_state.model_to_delete = model_file
                    st.rerun()

        if st.session_state.model_to_rename:
            self._draw_rename_ui()

        if st.session_state.model_to_delete:
            self._draw_delete_confirmation_ui()

        return os.path.join(models_dir, selected_model_name + ".pt") if selected_model_name else None

    def _draw_rename_ui(self):
        with st.form("rename_form"):
            st.info(f"Renaming: **{st.session_state.model_to_rename}**")
            new_name = st.text_input("Enter new name (without .pt extension)", value=st.session_state.new_model_name)

            submitted = st.form_submit_button("Confirm Rename")
            if submitted:
                old_path = os.path.join("Models", st.session_state.model_to_rename)
                new_path = os.path.join("Models", new_name + ".pt")
                if not new_name:
                    st.error("Name cannot be empty.")
                elif os.path.exists(new_path):
                    st.error(f"A model named '{new_name}.pt' already exists.")
                else:
                    os.rename(old_path, new_path)
                    st.session_state.model_to_rename = None
                    st.success(f"Renamed to {new_name}.pt")
                    st.rerun()

    def _draw_delete_confirmation_ui(self):
        st.error(
            f"Are you sure you want to delete **{st.session_state.model_to_delete}**? This action cannot be undone.")
        col1, col2 = st.columns(2)
        if col1.button("YES, DELETE IT", use_container_width=True, type="primary"):
            try:
                os.remove(os.path.join("Models", st.session_state.model_to_delete))
                st.success(f"Deleted {st.session_state.model_to_delete}")
                st.session_state.model_to_delete = None
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting file: {e}")

        if col2.button("Cancel", use_container_width=True):
            st.session_state.model_to_delete = None
            st.rerun()