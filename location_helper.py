#!/usr/bin/env python3
# location_helper.py

import googlemaps
import os
import logging
import maidenhead
from constants import GOOGLE_PLACES_API_KEY_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_google_places_api_key() -> str:
    """
    Loads the Google Places API key from the specified file.
    """
    if not os.path.exists(GOOGLE_PLACES_API_KEY_FILE):
        raise FileNotFoundError(f"API key file not found: {GOOGLE_PLACES_API_KEY_FILE}")
    with open(GOOGLE_PLACES_API_KEY_FILE, 'r') as f:
        api_key = f.read().strip()
    if not api_key:
        raise ValueError("Google Places API key file is empty.")
    return api_key

def get_gps_coordinates(location_description: str) -> str:
    """
    Gets GPS coordinates for a natural language location description.

    Args:
        location_description: A description of the location (e.g., "Eiffel Tower", "downtown San Francisco").

    Returns:
        A string with the latitude and longitude or an error message.
    """
    if not location_description:
        return "I need a location description to find the GPS coordinates."

    try:
        api_key = load_google_places_api_key()
        gmaps = googlemaps.Client(key=api_key)

        logging.info(f"Geocoding location: {location_description}")
        geocode_result = gmaps.geocode(location_description)

        if not geocode_result:
            logging.warning(f"No geocoding results for '{location_description}'.")
            return f"Sorry, I could not find GPS coordinates for '{location_description}'. Please be more specific or try a different location."

        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
        
        formatted_address = geocode_result[0]['formatted_address']
        
        # Convert to Maidenhead grid
        mh = maidenhead.to_maiden(lat, lng)
        
        response_text = f"The GPS coordinates for {formatted_address} are: Latitude {lat:.6f}, Longitude {lng:.6f}. The Maidenhead grid square is {mh}."
        logging.info(response_text)
        return response_text

    except FileNotFoundError as e:
        logging.error(f"Google Places API key error: {e}")
        return "The Google Places API key file is missing. Please configure it."
    except ValueError as e:
        logging.error(f"Google Places API key error: {e}")
        return "The Google Places API key is invalid. Please check the configuration."
    except Exception as e:
        logging.error(f"An unexpected error occurred during geocoding: {e}")
        return "An unexpected error occurred while trying to find the GPS coordinates."

if __name__ == '__main__':
    # Make sure to have a 'google_places_api_key.txt' file with a valid key for testing
    print("--- Testing valid location: 'Golden Gate Bridge' ---")
    print(get_gps_coordinates("Golden Gate Bridge"))

    print("\n--- Testing invalid location: 'asdfghjkl' ---")
    print(get_gps_coordinates("asdfghjkl"))

    print("\n--- Testing empty location ---")
    print(get_gps_coordinates("")) 