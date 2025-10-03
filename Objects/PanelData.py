import json


class PanelData:
    """
    A data class that holds the panel data extracted by the OCRProcessor
    in a structured way and provides easy access methods.
    This version is fully compatible with the new parsing logic.
    """

    def __init__(self, data_dict):
        """
        Initializes the PanelData object with the dictionary data from the OCR process.

        Args:
            data_dict (dict): The dictionary produced by OCRProcessor._parse_ocr_results.
        """
        # Keys are updated according to the new data structure
        self.flight_info = data_dict.get('flight_info', {})
        self.aircraft_details = data_dict.get('aircraft_details', {})
        self.unassigned_texts = data_dict.get('unassigned_texts', [])
        self.raw_data = data_dict

    # --- New Getter Methods for Flight Information ---
    def get_flight_number(self):
        """Returns the flight number (e.g., 'VY1872')."""
        return self.flight_info.get('flight_number')

    def get_airline(self):
        """Returns the name of the airline."""
        return self.flight_info.get('airline')

    def get_departure_code(self):
        """Returns the IATA code of the departure airport (e.g., 'BCN')."""
        return self.flight_info.get('departure_code')

    def get_arrival_code(self):
        """Returns the IATA code of the arrival airport (e.g., 'CPH')."""
        return self.flight_info.get('arrival_code')

    def get_departure_city(self):
        """Returns the departure city."""
        return self.flight_info.get('departure_city')

    def get_arrival_city(self):
        """Returns the arrival city."""
        return self.flight_info.get('arrival_city')

    # --- Getter Methods for Aircraft Details ---
    def get_registration(self):
        """Returns the aircraft's registration code (e.g., 'EC-NLV')."""
        return self.aircraft_details.get('registration')

    def get_aircraft_type(self):
        """Returns the aircraft's type."""
        return self.aircraft_details.get('type')

    def get_country_of_registration(self):
        """Returns the country of registration."""
        return self.aircraft_details.get('country_of_reg')

    def get_aircraft_category(self):
        """Returns the aircraft's category (e.g., 'Passenger')."""
        return self.aircraft_details.get('category')

    # --- Utility Methods for Compatibility and Output ---
    def to_json(self):
        """Returns all data as a string in JSON format."""
        return json.dumps(self.raw_data, indent=4, ensure_ascii=False)

    def __repr__(self):
        """Provides a more understandable text representation of the object."""
        flight_num = self.get_flight_number() or "N/A"
        reg = self.get_registration() or "N/A"
        return f"<PanelData flight='{flight_num}', registration='{reg}'>"