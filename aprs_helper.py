#!/usr/bin/env python3
# aprs_helper.py - APRS Message Utility
#
# This module provides functions for sending and receiving APRS messages
# by scraping the findu.com website. It is used by the W6RGC/AI Ham
# Radio Voice Assistant to enable APRS communication via voice commands.
#
# Key Features:
#  - Send APRS messages via findu.com
#  - Retrieve APRS messages from findu.com
#  - Parse HTML to extract message content
#  - Format messages for natural language presentation
#
# Usage:
#  from helper_aprs import send_aprs_message, get_aprs_messages
#  send_aprs_message("SENDER_CALL", "RECEIVER_CALL", "Hello from AI assistant!")
#  messages = get_aprs_messages("YOUR_CALLSIGN")
#  print(messages)
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
#

import requests
from bs4 import BeautifulSoup

# Send message HTTP GET command using findu.com
SEND_MESSAGE_URL = "http://www.findu.com/cgi-bin/sendmsg.cgi?fromcall={sender}&tocall={receiver}&msg={message}"

# View message HTTP GET command using findu.com
VIEW_MESSAGE_URL = "http://findu.com/cgi-bin/msg.cgi?call={receiver}"

APRS_MESSAGE_MAX_LENGTH = 50

# For preprocessing messages for Text-to-Speech
PHONETIC_ALPHABET = {
  "A": "Alpha",
  "B": "Bravo",
  "C": "Charlie",
  "D": "Delta",
  "E": "Echo",
  "F": "Foxtrot",
  "G": "Golf",
  "H": "Hotel",
  "I": "India",
  "J": "Juliett",
  "K": "Kilo",
  "L": "Lima",
  "M": "Mike",
  "N": "November",
  "O": "Oscar",
  "P": "Papa",
  "Q": "Quebec",
  "R": "Romeo",
  "S": "Sierra",
  "T": "Tango",
  "U": "Uniform",
  "V": "Victor",
  "W": "Whiskey",
  "X": "X-ray",
  "Y": "Yankee",
  "Z": "Zulu",
  "0": "Zero",
  "1": "One",
  "2": "Two",
  "3": "Three",
  "4": "Four",
  "5": "Five",
  "6": "Six",
  "7": "Seven",
  "8": "Eight",
  "9": "Nine",
  "-": "Dash",
  "/": "Stroke"
}

def send_aprs_message(sender, receiver, message):
    url = SEND_MESSAGE_URL.format(sender=sender, receiver=receiver, message=message)
    # Force the message to be less than or equal to APRS_MESSAGE_MAX_LENGTH characters
    message = message[:APRS_MESSAGE_MAX_LENGTH]
    response = requests.get(url)
    return response.text

def get_aprs_messages(receiver):
    print(f"Getting APRS messages for {receiver}")
    url = VIEW_MESSAGE_URL.format(receiver=receiver)
    response = requests.get(url)
    messages = _parse_aprs_messages(response.text)
    return _natural_language_messages(messages)

def _parse_aprs_messages(html_content):
    """
    Parses HTML content to extract APRS messages into a list of dictionaries.

    Args:
        html_content: A string containing the HTML document.

    Returns:
        A list of dictionaries, where each dictionary represents a message.
    """
    # Create a BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the main table containing the messages
    message_table = soup.find('table')
    
    # Initialize an empty list to store the parsed messages
    messages_list = []
    
    if not message_table:
        return messages_list

    # Find all message rows. In this HTML, they have a specific background color.
    # We skip the first row (the header) by selecting only rows with this attribute.
    message_rows = message_table.find_all('tr', bgcolor='#ccffcc')
    
    # Loop through each message row
    for row in message_rows:
        # Get all the cells (<td>) in the current row
        cells = row.find_all('td')
        
        # Ensure the row has the expected number of cells (5 in this case)
        if len(cells) >= 5:
            # Extract the text from each relevant cell and strip whitespace
            from_callsign = cells[0].get_text(strip=True)
            to_callsign = cells[1].get_text(strip=True)
            message_text = cells[4].get_text(strip=True)
            
            # Create a dictionary for the message
            message_data = {
                "From": from_callsign,
                "To": to_callsign,
                "Message": message_text
            }
            
            # Add the dictionary to our list
            messages_list.append(message_data)
            
    return messages_list

def _format_message(message):
    # Split at '{' if present and take only the first part
    if '{' in message:
        message = message.split('{')[0]
    
    # Handle @ followed by numbers
    if '@' in message:
        parts = message.split('@')
        for i in range(1, len(parts)):
            if parts[i][0].isdigit():
                # Add spaces between numbers
                numbers = ' '.join(parts[i][0])
                parts[i] = numbers + parts[i][1:]
        message = '@'.join(parts)
    
    return message

def _natural_language_messages(messages):
    natural_language_response = f"Here are your last {len(messages)} Automatic Packet Reporting System messages.\n"
    for i, msg in enumerate(messages[:5], 1):
         natural_language_response += f"""
                Message {i}, from {_phonetic_alphabet(msg['From'])}.
                The message is {_format_message(msg['Message'])}.\n"""
    natural_language_response += "End of messages"
    return natural_language_response

def _phonetic_alphabet(input_string):
  return_string = ""
  for char in input_string:
    if char in PHONETIC_ALPHABET:
      return_string += PHONETIC_ALPHABET[char] + " "
    else:
      return_string += char + " "
  return return_string