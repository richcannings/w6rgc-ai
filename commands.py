# W6RGC-AI: Off Grid Ham Radio AI Voice Assistant
#
# Author: Rich Cannings <rcannings@gmail.com>
# License: Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
#
# Description:
# This module handles the identification and parsing of voice commands
# for the W6RGC-AI voice assistant. It checks for specific keywords
# to trigger actions like termination, status reports, chat resets,
# and identification.
#
import re
from constants import MAX_COMMAND_WORDS, BOT_NAME

def handle_command(operator_text):
    """
    Identifies known commands from the operator_text.
    Only checks the first MAX_COMMAND_WORDS words to prevent commands from being triggered later in conversations.

    Args:
        operator_text (str): The transcribed text from the operator.

    Returns:
        str: Command type ("terminate", "status", "reset", "identify") or None if no command is matched.
    """
    # bot_name is now the directly imported constant BOT_NAME
    
    # Only check the first MAX_COMMAND_WORDS words to prevent accidental command triggers in conversations
    words = operator_text.split()
    first_words = ' '.join(words[:MAX_COMMAND_WORDS])
    print(f"🔍 Checking for commands in first {MAX_COMMAND_WORDS} words: '{first_words}'")
    
    # Use the truncated text for command detection
    text_to_check = first_words

    # Check for termination command
    if re.search(rf"{re.escape(BOT_NAME)}.*?\b(break|brake|exit|quit|shutdown)\b", text_to_check, re.IGNORECASE):
        print("🛑 Termination command detected by commands.py.")
        return "terminate"

    # Check for status command
    if re.search(rf"{re.escape(BOT_NAME)}.*?\b(status|report)\b", text_to_check, re.IGNORECASE):
        print("⚙️ Status command detected by commands.py.")
        return "status"

    # Check for reset/new chat command
    if re.search(rf"{re.escape(BOT_NAME)}.*?\b(reset|start a new chat|new chat)\b", text_to_check, re.IGNORECASE):
        print("🔄 Reset command detected by commands.py.")
        return "reset"

    # Check for identify command
    if re.search(rf"{re.escape(BOT_NAME)}.*?\b(identify)\b", text_to_check, re.IGNORECASE) or \
       re.search(r"\b(identify|call sign|what is your call sign|who are you)\b", text_to_check, re.IGNORECASE):
        print("🆔 Identify command detected by commands.py.")
        return "identify"
        
    return None 