#!/usr/bin/env python3
# llm_ollama.py - Ollama Local LLM Integration
#
# This module provides integration with local Ollama LLM instances for the 
# W6RGC/AI Ham Radio Voice Assistant. It offers offline AI capabilities
# without requiring internet connectivity.
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

import json
import requests
from constants import DEFAULT_OFFLINE_MODEL, OLLAMA_URL

requires_internet = False

def convert_ollama_response(response_text):
    """
    Converts the JSON response from Ollama to a string.
    """
    try:
        # Try to parse the response as JSON
        response_list = json.loads(response_text)
        if isinstance(response_list, list):
            # Join all sentences with a space
            return ' '.join(response_list)
        return response_text
    except json.JSONDecodeError:
        # If not valid JSON, return the original text
        return response_text

def ask_ollama(prompt):
    """
    Asks Ollama for a response to the given prompt.
    """
    payload = {
        "model": DEFAULT_OFFLINE_MODEL,
        "prompt": prompt
    }
    response = requests.post(OLLAMA_URL, json=payload, stream=True)
    response.raise_for_status()
    result = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            result += data.get("response", "")
    return result