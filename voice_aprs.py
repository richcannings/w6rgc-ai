# voice_aprs.py - scrape findu.com for APRS message sending and receiving

import requests
from bs4 import BeautifulSoup

# Send message HTTP GET command using findu.com
SEND_MESSAGE_URL= "http://www.findu.com/cgi-bin/sendmsg.cgi?fromcall={sender}&tocall={receiver}&msg={message}"

# View message HTTP GET command using findu.com
VIEW_MESSAGE_URL= "http://findu.com/cgi-bin/msg.cgi?call={receiver}"

APRS_MESSAGE_MAX_LENGTH= 50

def send_aprs_message(sender, receiver, message):
    url = SEND_MESSAGE_URL.format(sender=sender, receiver=receiver, message=message)
    # Force the message to be less than or equal to APRS_MESSAGE_MAX_LENGTH
    #  characters
    message = message[:APRS_MESSAGE_MAX_LENGTH]
    response = requests.get(url)
    return response.text

def get_aprs_messages(receiver):
    url = VIEW_MESSAGE_URL.format(receiver=receiver)
    response = requests.get(url)
    messages = _parse_aprs_messages(response.text)
    return natural_language_messages(messages)

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
        if len(cells) == 5:
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

def natural_language_messages(messages):
    natural_language_response = ""
    for i, msg in enumerate(messages, 1):
         natural_language_response += f"({i}) From: {msg['From']}, To: {msg['To']}, Message: {msg['Message']}\n"
    return natural_language_response