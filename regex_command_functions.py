#!/usr/bin/env python3
# regex_command_functions.py - Voice Command Handling
#
# This module handles the identification and parsing of voice commands
# for the W6RGC-AI voice assistant. It checks for specific keywords
# to trigger actions like termination, status reports, chat resets,
# and identification.
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
        print("🛑 Termination command detected by regex_command_functions.py.")
        return "terminate"

    # Check for status command
    if re.search(rf"{re.escape(BOT_NAME)}.*?\b(status)\b", text_to_check, re.IGNORECASE):
        print("⚙️ Status command detected by regex_command_functions.py.")
        return "status"

    # Check for reset/new chat command
    if re.search(rf"{re.escape(BOT_NAME)}.*?\b(reset|start a new chat|new chat)\b", text_to_check, re.IGNORECASE):
        print("🔄 Reset command detected by regex_command_functions.py.")
        return "reset"

    # Check for identify command
    if re.search(rf"{re.escape(BOT_NAME)}.*?\b(identify)\b", text_to_check, re.IGNORECASE) or \
       re.search(r"\b(identify|call sign|what is your call sign|who are you)\b", text_to_check, re.IGNORECASE):
        print("🆔 Identify command detected by regex_command_functions.py.")
        return "identify"
        
    return None

# Test section for standalone execution
if __name__ == "__main__":
    print("🧪 Testing regex_command_functions.py")
    print(f"Using BOT_NAME: {BOT_NAME}")
    print(f"Checking first {MAX_COMMAND_WORDS} words for commands")
    
    # Test cases
    test_cases = [
        f"{BOT_NAME} break",
        f"{BOT_NAME} exit now",
        f"{BOT_NAME} status report",
        f"{BOT_NAME} reset the chat",
        f"{BOT_NAME} identify yourself",
        "identify please",
        "what is your call sign",
        f"{BOT_NAME} this is just a normal conversation that should not trigger commands",
        "this is a long conversation where the word break appears later in the sentence"
    ]
    
    for test_text in test_cases:
        print(f"\n📝 Testing: '{test_text}'")
        result = handle_command(test_text)
        if result:
            print(f"✅ Command detected: {result}")
        else:
            print("❌ No command detected")
    
    print("\n�� Test complete!") 