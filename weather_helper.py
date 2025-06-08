#!/usr/bin/env python3
# weather_helper.py - Weather Information Utility
#
# This module provides functions for retrieving current weather and forecast
# information using the OpenWeatherMap API. It is used by the W6RGC/AI Ham
# Radio Voice Assistant to provide weather information via voice commands.
#
# Key Features:
#  - Get current weather conditions
#  - Retrieve 3-day weather forecast
#  - Format weather data for natural language presentation
#  - Handle API errors gracefully
#
# Usage:
#  from weather_helper import get_weather_forecast
#  weather_info = get_weather_forecast("San Francisco, CA")
#  print(weather_info)
#
# Author: Rich Cannings <rcannings@gmail.com>
# Copyright 2025 Rich Cannings
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# OpenWeatherMap API endpoints
CURRENT_WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "http://api.openweathermap.org/data/2.5/forecast"
GEOCODING_URL = "http://api.openweathermap.org/geo/1.0/direct"

# You'll need to get a free API key from openweathermap.org
WEATHER_API_KEY_FILE = "weather_api_key.txt"

def load_weather_api_key() -> Optional[str]:
    """
    Load the OpenWeatherMap API key from the weather_api_key.txt file.
    
    Returns:
        str: The API key, or None if not found
    """
    try:
        with open(WEATHER_API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        return api_key if api_key else None
    except FileNotFoundError:
        print(f"⚠️  Weather API key file '{WEATHER_API_KEY_FILE}' not found")
        return None
    except Exception as e:
        print(f"❌ Error reading weather API key file: {e}")
        return None

def get_coordinates(location: str) -> Optional[tuple]:
    """
    Get latitude and longitude for a given location using OpenWeatherMap geocoding.
    
    Args:
        location (str): Location name (e.g., "San Francisco, CA" or "New York")
        
    Returns:
        tuple: (latitude, longitude) or None if not found
    """
    api_key = load_weather_api_key()
    if not api_key:
        print("❌ No API key available for geocoding")
        return None
    
    # Try different variations of the location name
    location_variations = [
        location,  # Original location
        location.split(',')[0].strip() if ',' in location else None,  # Just the city name
        location.replace(', ', ',') if ', ' in location else None,  # Remove spaces after commas
        location.replace(',', '') if ',' in location else None,    # Remove commas entirely
    ]
    
    # Remove None values and duplicates while preserving order
    seen = set()
    unique_variations = []
    for loc in location_variations:
        if loc and loc not in seen:
            seen.add(loc)
            unique_variations.append(loc)
    
    for variation in unique_variations:
        try:
            params = {
                'q': variation,
                'limit': 1,
                'appid': api_key
            }
            
            response = requests.get(GEOCODING_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data:
                coords = (data[0]['lat'], data[0]['lon'])
                return coords
        
        except Exception as e:
            print(f"❌ Error getting coordinates for '{variation}': {e}")
            continue
    
    return None

def get_current_weather_by_coords(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Get current weather conditions for specific coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        dict: Weather data or None if error
    """
    api_key = load_weather_api_key()
    if not api_key:
        return None
    
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'imperial'  # Use Fahrenheit
        }
        
        response = requests.get(CURRENT_WEATHER_URL, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"❌ Error getting current weather for coordinates ({lat}, {lon}): {e}")
        return None

def get_forecast_by_coords(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Get 5-day weather forecast for specific coordinates.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        dict: Forecast data or None if error
    """
    api_key = load_weather_api_key()
    if not api_key:
        return None
    
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'imperial'  # Use Fahrenheit
        }
        
        response = requests.get(FORECAST_URL, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        print(f"❌ Error getting forecast for coordinates ({lat}, {lon}): {e}")
        return None

def get_current_weather(location: str) -> Optional[Dict[str, Any]]:
    """
    Get current weather conditions for a location.
    
    Args:
        location (str): Location name
        
    Returns:
        dict: Weather data or None if error
    """
    # Get coordinates first
    coords = get_coordinates(location)
    if not coords:
        return None
    
    lat, lon = coords
    return get_current_weather_by_coords(lat, lon)

def get_forecast(location: str) -> Optional[Dict[str, Any]]:
    """
    Get 5-day weather forecast for a location (we'll use first 3 days).
    
    Args:
        location (str): Location name
        
    Returns:
        dict: Forecast data or None if error
    """
    # Get coordinates first
    coords = get_coordinates(location)
    if not coords:
        return None
    
    lat, lon = coords
    return get_forecast_by_coords(lat, lon)

def format_weather_for_speech(current_weather: Dict[str, Any], forecast_data: Dict[str, Any]) -> str:
    """
    Format weather data for natural language presentation.
    
    Args:
        current_weather: Current weather data from API
        forecast_data: Forecast data from API
        
    Returns:
        str: Formatted weather description for TTS
    """
    try:
        # Current weather
        current_temp = round(current_weather['main']['temp'])
        current_desc = current_weather['weather'][0]['description']
        current_humidity = current_weather['main']['humidity']
        feels_like = round(current_weather['main']['feels_like'])
        
        location_name = current_weather['name']
        
        weather_text = f"Current weather in {location_name}: {current_temp} degrees Fahrenheit, {current_desc}. "
        weather_text += f"Feels like {feels_like} degrees. Humidity is {current_humidity} percent.\n\n"
        
        # 3-day forecast
        weather_text += "3-day forecast:\n"
        
        # Group forecast data by date
        daily_forecasts = {}
        for item in forecast_data['list'][:24]:  # Get next 24 forecasts (3 days worth)
            date = datetime.fromtimestamp(item['dt']).date()
            if date not in daily_forecasts:
                daily_forecasts[date] = []
            daily_forecasts[date].append(item)
        
        # Get first 3 days
        forecast_dates = sorted(daily_forecasts.keys())[:3]
        
        for i, date in enumerate(forecast_dates, 1):
            day_data = daily_forecasts[date]
            
            # Calculate daily high/low
            temps = [item['main']['temp'] for item in day_data]
            high_temp = round(max(temps))
            low_temp = round(min(temps))
            
            # Get most common weather condition
            conditions = [item['weather'][0]['description'] for item in day_data]
            main_condition = max(set(conditions), key=conditions.count)
            
            # Calculate chance of rain
            rain_chance = 0
            for item in day_data:
                if 'pop' in item:  # Probability of precipitation
                    rain_chance = max(rain_chance, item['pop'])
            rain_chance = round(rain_chance * 100)
            
            # Format day name
            if i == 1:
                day_name = "Today" if date == datetime.now().date() else "Tomorrow"
            else:
                day_name = date.strftime("%A")
            
            weather_text += f"{day_name}: High {high_temp}, low {low_temp} degrees. "
            weather_text += f"{main_condition.capitalize()}. "
            weather_text += f"Chance of rain: {rain_chance} percent.\n"
        
        return weather_text.strip()
        
    except Exception as e:
        print(f"❌ Error formatting weather data: {e}")
        return "Sorry, I couldn't format the weather information properly."

def get_weather_forecast(location: str) -> str:
    """
    Get complete weather information (current + 3-day forecast) for a location.
    
    Args:
        location (str): Location name (e.g., "San Francisco, CA")
        
    Returns:
        str: Formatted weather information for TTS
    """
    # Check if we have an API key
    if not load_weather_api_key():
        return "Weather service is not available. Please add your OpenWeatherMap API key to weather_api_key.txt file."
    
    print(f"🌤️  Getting weather for {location}...")
    
    # Get coordinates first (only once)
    coords = get_coordinates(location)
    if not coords:
        return f"Sorry, I couldn't find the location '{location}'. Please check if the location name is correct."
    
    lat, lon = coords
    
    # Get current weather using coordinates
    current_weather = get_current_weather_by_coords(lat, lon)
    if not current_weather:
        return f"Sorry, I couldn't get weather information for {location}. Please try again later."
    
    # Get forecast using same coordinates
    forecast_data = get_forecast_by_coords(lat, lon)
    if not forecast_data:
        return f"Sorry, I couldn't get the weather forecast for {location}. Please try again later."
    
    # Format for speech
    return format_weather_for_speech(current_weather, forecast_data)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing weather functionality...")
    
    test_location = "San Francisco, CA"
    weather_info = get_weather_forecast(test_location)
    print(f"\n🌤️  Weather for {test_location}:")
    print(weather_info) 