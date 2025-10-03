import config
from fpdf import FPDF
from datetime import datetime
import copy


class Report:
    """
    Records the history of all objects detected during the video processing
    and generates a detailed PDF report at the end of the process.
    """

    def __init__(self):
        """
        Initializes the reporter with an empty dictionary to hold the history data.
        """
        # Main data structure for storing history data:
        # { aircraft_id: {'static_data': {...}, 'timeline': [...]}, ... }
        self.history = {}
        self.current_frame_number = 0

    def log_frame_data(self):
        """
        Called after each frame is processed.
        It takes the instantaneous aircraft data from the config and adds it to its own history record.
        """
        self.current_frame_number += config.skip_frame

        # Get all current aircraft from config
        all_aircrafts = config.aircraft_manager.get_all_aircrafts()

        for aircraft in all_aircrafts:
            aircraft_id = aircraft.id

            # If this aircraft is seen for the first time, create its history record
            if aircraft_id not in self.history:
                self.history[aircraft_id] = {
                    'static_data': {
                        'cls_id': aircraft.cls_id,
                        'panel_data': None  # Initially no panel data
                    },
                    'timeline': []
                }

            # If panel data is assigned to the aircraft and we don't have it recorded, save it
            if aircraft.panel is not None and self.history[aircraft_id]['static_data']['panel_data'] is None:
                # Store by copying the PanelData object
                self.history[aircraft_id]['static_data']['panel_data'] = copy.deepcopy(aircraft.panel)

            # Create the state of the current frame as a dictionary
            frame_state = {
                'frame': self.current_frame_number,
                'location': aircraft.location,
                'direction': aircraft.direction,
                'velocity': aircraft.velocity,
                'condition': aircraft.condition
            }

            # Add this state to the aircraft's timeline
            self.history[aircraft_id]['timeline'].append(frame_state)

    def generate_pdf_report(self):
        """
        Generates and saves the PDF report using all the collected history data.
        """
        if not self.history:
            print("No logged data found to generate a report.")
            return

        print("Generating PDF report...")
        pdf = FPDF()
        # It is important to add a font to avoid issues with non-ASCII characters.
        # Make sure this font file (.ttf) is in your project's directory.
        try:
            pdf.add_font('DejaVu', '', 'Font/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed.ttf', uni=True)
            pdf.add_font('DejaVu', 'B', 'Font/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed-Bold.ttf', uni=True)
            pdf.add_font('DejaVu', 'I', 'Font/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed-Oblique.ttf', uni=True)
            pdf.add_font('DejaVu', 'BI', 'Font/dejavu-fonts-ttf-2.37/ttf/DejaVuSansCondensed-BoldOblique.ttf', uni=True)
            font_family = 'DejaVu'
        except RuntimeError:
            print("Warning: DejaVu font not found. Using standard font (non-ASCII characters may cause issues).")
            font_family = 'Arial'

        # Create a separate section for each aircraft
        for aircraft_id, data in self.history.items():
            pdf.add_page()

            # --- Header Section ---
            static_info = data['static_data']
            cls_name = config.cls_names.get(static_info['cls_id'], "Unknown Class")
            pdf.set_font(font_family, 'B', 16)
            pdf.cell(0, 10, f'Tracking Report: Object ID {aircraft_id} ({cls_name})', 0, 1, 'C')
            pdf.ln(5)

            # --- Panel Data Section (if available) ---
            panel_data = static_info.get('panel_data')
            if panel_data is not None:
                pdf.set_font(font_family, 'B', 12)
                pdf.cell(0, 10, 'Flight Information (Panel Data)', 0, 1, 'L')
                pdf.set_font(font_family, '', 10)

                # MODIFICATION: Replaced calls to deprecated methods (get_ground_speed, get_barometric_altitude)
                # with new methods that reflect the updated data structure from the parser.

                # Retrieve all relevant data using the new PanelData methods
                flight_num = panel_data.get_flight_number() or "N/A"
                airline = panel_data.get_airline() or "N/A"
                reg = panel_data.get_registration() or "N/A"
                ac_type = panel_data.get_aircraft_type() or "N/A"
                dep_code = panel_data.get_departure_code() or "???"
                arr_code = panel_data.get_arrival_code() or "???"
                route = f"{dep_code} to {arr_code}"

                # Format the information into a multi-line string for better readability in the PDF
                info_text = (
                    f"Flight: {flight_num} ({airline})\n"
                    f"Route: {route}\n"
                    f"Registration: {reg}   |   Aircraft Type: {ac_type}"
                )

                # Use multi_cell to render the multi-line string with a border
                pdf.multi_cell(0, 6, info_text, border=1, align='L')
                pdf.ln(10)

            # CHANGE: Table header updated (Speed removed, Location header changed)
            pdf.set_font(font_family, 'B', 10)
            col_widths = {'frame': 30, 'location': 65, 'direction': 45, 'condition': 50}
            pdf.cell(col_widths['frame'], 8, 'Frame No', 1, 0, 'C')
            pdf.cell(col_widths['location'], 8, 'Location', 1, 0, 'C')
            pdf.cell(col_widths['direction'], 8, 'Direction (Angle)', 1, 0, 'C')
            pdf.cell(col_widths['condition'], 8, 'Status', 1, 1, 'C')

            # CHANGE: Table content updated (velocity_str removed)
            pdf.set_font(font_family, '', 9)
            for frame_log in data['timeline']:
                try:
                    condition_str = config.conditions[frame_log['condition']]
                except (IndexError, TypeError):
                    condition_str = "Unknown"

                direction_str = f"{frame_log['direction']:.1f}°" if frame_log['direction'] is not None else "N/A"

                pdf.cell(col_widths['frame'], 8, str(frame_log['frame']), 1, 0, 'C')
                pdf.cell(col_widths['location'], 8, str(frame_log['location']), 1, 0, 'C')
                pdf.cell(col_widths['direction'], 8, direction_str, 1, 0, 'C')
                pdf.cell(col_widths['condition'], 8, condition_str, 1, 1, 'C')

        # The report date is added to the last page
        pdf.set_font(font_family, 'I', 8)
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.set_y(-15)
        pdf.cell(0, 10, f'Report Generation Date: {report_time}', 0, 0, 'C')

        try:
            pdf.output(config.pdf_report_path)
            print(f"PDF report successfully saved to '{config.pdf_report_path}'.")
        except Exception as e:
            print(f"ERROR: An issue occurred while saving the PDF report: {e}")