#!/usr/bin/env python3
# time_helper.py - Time Zone Information Utility
#
# This module provides functions for retrieving current time and date
# information for various time zones. It is used by the W6RGC/AI Ham
# Radio Voice Assistant to provide time information via voice commands.
#
# Key Features:
#  - Get current time in any time zone
#  - Natural language time zone parsing
#  - Default to Pacific Time if no zone specified
#  - Format output for natural language presentation
#  - Handle common time zone abbreviations and names
#
# Usage:
#  from time_helper import get_timezone_time
#  time_info = get_timezone_time("Eastern")
#  print(time_info)
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

from datetime import datetime
import pytz
from typing import Optional, Dict

# Default timezone if none specified
DEFAULT_TIMEZONE = "US/Pacific"

# Common timezone mappings for natural language processing
TIMEZONE_MAPPINGS = {
    # US Time Zones
    "pacific": "US/Pacific",
    "pacific time": "US/Pacific",
    "pt": "US/Pacific",
    "pdt": "US/Pacific",
    "pst": "US/Pacific",
    "west coast": "US/Pacific",
    "california": "US/Pacific",
    "san francisco": "US/Pacific",
    "los angeles": "US/Pacific",
    "seattle": "US/Pacific",
    
    "mountain": "US/Mountain",
    "mountain time": "US/Mountain",
    "mt": "US/Mountain",
    "mdt": "US/Mountain",
    "mst": "US/Mountain",
    "denver": "US/Mountain",
    "colorado": "US/Mountain",
    "arizona": "US/Arizona",
    "phoenix": "US/Arizona",
    
    "central": "US/Central",
    "central time": "US/Central",
    "ct": "US/Central",
    "cdt": "US/Central",
    "cst": "US/Central",
    "chicago": "US/Central",
    "texas": "US/Central",
    "dallas": "US/Central",
    "houston": "US/Central",
    
    "eastern": "US/Eastern",
    "eastern time": "US/Eastern",
    "et": "US/Eastern",
    "edt": "US/Eastern",
    "est": "US/Eastern",
    "east coast": "US/Eastern",
    "new york": "US/Eastern",
    "boston": "US/Eastern",
    "atlanta": "US/Eastern",
    "miami": "US/Eastern",
    
    # International
    "utc": "UTC",
    "gmt": "GMT",
    "greenwich": "GMT",
    "zulu": "UTC",
    "z": "UTC",
    
    "london": "Europe/London",
    "uk": "Europe/London",
    "britain": "Europe/London",
    "england": "Europe/London",
    
    "paris": "Europe/Paris",
    "france": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "germany": "Europe/Berlin",
    "rome": "Europe/Rome",
    "italy": "Europe/Rome",
    
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "jst": "Asia/Tokyo",
    
    "sydney": "Australia/Sydney",
    "australia": "Australia/Sydney",
    
    "hawaii": "US/Hawaii",
    "honolulu": "US/Hawaii",
    "hst": "US/Hawaii",
    
    "alaska": "US/Alaska",
    "anchorage": "US/Alaska",
    "akst": "US/Alaska",
    "akdt": "US/Alaska",
}

def parse_timezone(timezone_input: str) -> str:
    """
    Parse natural language timezone input into a pytz timezone string.
    
    Args:
        timezone_input (str): Natural language timezone description
        
    Returns:
        str: pytz timezone string
    """
    if not timezone_input:
        return DEFAULT_TIMEZONE
    
    # Normalize input
    normalized = timezone_input.lower().strip()
    
    # Check direct mappings first
    if normalized in TIMEZONE_MAPPINGS:
        return TIMEZONE_MAPPINGS[normalized]
    
    # Check if it's already a valid pytz timezone
    try:
        pytz.timezone(timezone_input)
        return timezone_input
    except pytz.exceptions.UnknownTimeZoneError:
        pass
    
    # Try partial matches for city names or regions (but avoid short matches)
    for key, value in TIMEZONE_MAPPINGS.items():
        if len(key) >= 4:  # Only match longer strings to avoid false positives
            if key in normalized or normalized in key:
                return value
    
    # Default to Pacific if we can't parse it
    print(f"⚠️  Unknown timezone '{timezone_input}', defaulting to Pacific Time")
    return DEFAULT_TIMEZONE

def get_timezone_time(timezone_input: Optional[str] = None) -> str:
    """
    Get current time and date for a specified timezone.
    
    Args:
        timezone_input (str, optional): Timezone description. Defaults to Pacific Time.
        
    Returns:
        str: Formatted time and date string for TTS
    """
    try:
        # Parse the timezone
        if timezone_input:
            tz_string = parse_timezone(timezone_input)
            print(f"🕒 Getting time for timezone: {timezone_input} -> {tz_string}")
        else:
            tz_string = DEFAULT_TIMEZONE
            print(f"🕒 Using default timezone: {tz_string}")
        
        # Get the timezone object
        try:
            tz = pytz.timezone(tz_string)
        except pytz.exceptions.UnknownTimeZoneError:
            print(f"❌ Invalid timezone: {tz_string}, falling back to Pacific")
            tz = pytz.timezone(DEFAULT_TIMEZONE)
            tz_string = DEFAULT_TIMEZONE
        
        # Get current time in the specified timezone
        now_utc = datetime.now(pytz.UTC)
        local_time = now_utc.astimezone(tz)
        
        # Format the time without seconds and year
        time_str = local_time.strftime("%I:%M %p")  # 12-hour format with AM/PM
        date_str = local_time.strftime("%A, %B %d")  # Day of week, Month DD
        
        # Get timezone name for display
        tz_name = get_timezone_display_name(tz_string, local_time)
        
        # Create natural language response
        response = f"The current time in {tz_name} is {time_str} on {date_str}."
        
        return response
        
    except Exception as e:
        print(f"❌ Error getting time for timezone '{timezone_input}': {e}")
        return f"Sorry, I couldn't get the time information. Error: {str(e)}"

def get_timezone_display_name(tz_string: str, local_time: datetime) -> str:
    """
    Get a user-friendly display name for a timezone.
    
    Args:
        tz_string (str): pytz timezone string
        local_time (datetime): Current time in that timezone
        
    Returns:
        str: User-friendly timezone name
    """
    # Map common timezone strings to friendly names
    friendly_names = {
        "US/Pacific": "Pacific Time",
        "US/Mountain": "Mountain Time", 
        "US/Central": "Central Time",
        "US/Eastern": "Eastern Time",
        "US/Hawaii": "Hawaii Time",
        "US/Alaska": "Alaska Time",
        "US/Arizona": "Arizona Time",
        "UTC": "UTC",
        "GMT": "Greenwich Mean Time",
        "Europe/London": "London Time",
        "Europe/Paris": "Paris Time",
        "Europe/Berlin": "Berlin Time",
        "Europe/Rome": "Rome Time",
        "Asia/Tokyo": "Tokyo Time",
        "Australia/Sydney": "Sydney Time",
    }
    
    if tz_string in friendly_names:
        return friendly_names[tz_string]
    
    # For other timezones, try to create a readable name
    if "/" in tz_string:
        parts = tz_string.split("/")
        if len(parts) >= 2:
            return f"{parts[-1].replace('_', ' ')} Time"
    
    # Fallback to the timezone abbreviation
    return local_time.strftime("%Z")

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing time zone functionality...")
    
    # Test various timezone inputs
    test_cases = [
        None,  # Default Pacific
        "Eastern",
        "New York", 
        "London",
        "Tokyo",
        "UTC",
        "Mountain Time",
        "invalid timezone"
    ]
    
    for test_case in test_cases:
        print(f"\n🕒 Testing: {test_case}")
        result = get_timezone_time(test_case)
        print(f"   Result: {result}") 