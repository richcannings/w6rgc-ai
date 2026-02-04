#!/usr/bin/env python3
# llm_chatgpt.py - OpenAI ChatGPT API Integration
#
# This module provides integration with OpenAI's ChatGPT API for the W6RGC/AI
# Ham Radio Voice Assistant.
#
# Key Features:
#  - Direct integration with OpenAI ChatGPT API
#  - Automatic API key loading from openai_api_key.txt
#  - Error handling and fallback responses
#  - Configurable model selection (via DEFAULT_CHATGPT_MODEL)
#
# Usage:
#  from llm_chatgpt import ask_chatgpt
#  response = ask_chatgpt("What is the weather like today?")
#
# Requirements:
#  - openai library (pip install openai)
#  - Valid OpenAI API key in openai_api_key.txt
#  - Internet connection for API access
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

import os
import time
from typing import Optional

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "openai is required. Install it with `pip install openai`."
    ) from exc

from constants import (
    OPENAI_API_KEY_FILE,
    DEFAULT_CHATGPT_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT
)


class ChatGPTAPIError(Exception):
    """Custom exception for ChatGPT API related errors."""
    pass


requires_internet = True


def load_api_key() -> str:
    """
    Load the OpenAI API key from the openai_api_key.txt file or environment.

    Returns:
        str: The API key
    """
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    if not os.path.exists(OPENAI_API_KEY_FILE):
        raise ChatGPTAPIError(f"API key file '{OPENAI_API_KEY_FILE}' not found")

    with open(OPENAI_API_KEY_FILE, 'r') as f:
        api_key = f.read().strip()

    if not api_key:
        raise ChatGPTAPIError(f"API key file '{OPENAI_API_KEY_FILE}' is empty")

    return api_key


def initialize_chatgpt() -> OpenAI:
    """
    Initialize the OpenAI client with the API key.
    """
    api_key = load_api_key()
    return OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)


def ask_chatgpt(prompt: str, model_name: Optional[str] = None) -> str:
    """
    Send a prompt to ChatGPT and return the response text.
    """
    if not prompt or not prompt.strip():
        raise ChatGPTAPIError("Prompt cannot be empty")

    client = initialize_chatgpt()
    model = model_name if model_name else DEFAULT_CHATGPT_MODEL

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"🤖 Sending prompt to ChatGPT (attempt {attempt + 1}/{MAX_RETRIES})...")
            response = client.responses.create(
                model=model,
                input=prompt
            )
            text = response.output_text.strip() if response and response.output_text else ""
            if text:
                print(f"✅ Received direct response from ChatGPT ({len(text)} characters)")
                return text
            raise ChatGPTAPIError("Empty response received from ChatGPT")
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️  ChatGPT API attempt {attempt + 1} failed: {exc}")
                print(f"🔄 Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                break

    raise ChatGPTAPIError(f"Failed to get response from ChatGPT. Last error: {last_error}")
