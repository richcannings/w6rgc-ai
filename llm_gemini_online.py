#!/usr/bin/env python3
# llm_gemini_online.py - Google Gemini API Integration
#
# This module provides integration with Google's Gemini API for the W6RGC/AI 
# Ham Radio Voice Assistant. It offers an alternative to the local Ollama LLM
# by connecting to Google's cloud-based Gemini models.
#
# Key Features:
#  - Direct integration with Google Gemini API
#  - Automatic API key loading from gemini_api_key.txt
#  - Error handling and fallback responses
#  - Configurable model selection (gemini-pro by default)
#  - Response streaming support for real-time applications
#
# Usage:
#  from llm_gemini_online import ask_gemini
#  response = ask_gemini("What is the weather like today?")
#
# Requirements:
#  - google-generativeai library (pip install google-generativeai)
#  - Valid Google AI API key in gemini_api_key.txt
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
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from typing import Optional, Dict, Any
import time

from constants import (
    GEMINI_API_KEY_FILE,
    DEFAULT_ONLINE_MODEL,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
    BOT_CALLSIGN
)
from aprs_helper import get_aprs_messages, send_aprs_message
from weather_helper import get_weather_forecast
from time_helper import get_timezone_time

class GeminiAPIError(Exception):
    """Custom exception for Gemini API related errors."""
    pass

def load_api_key() -> str:
    """
    Load the Gemini API key from the gemini_api_key.txt file.
    
    Returns:
        str: The API key
        
    Raises:
        GeminiAPIError: If the API key file is not found or empty
    """
    try:
        if not os.path.exists(GEMINI_API_KEY_FILE):
            raise GeminiAPIError(f"API key file '{GEMINI_API_KEY_FILE}' not found")
        
        with open(GEMINI_API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        
        if not api_key:
            raise GeminiAPIError(f"API key file '{GEMINI_API_KEY_FILE}' is empty")
        
        return api_key
    except Exception as e:
        if isinstance(e, GeminiAPIError):
            raise
        raise GeminiAPIError(f"Error reading API key file: {e}")

def initialize_gemini() -> genai.GenerativeModel:
    """
    Initialize the Gemini API client with the API key.
    
    Returns:
        genai.GenerativeModel: Configured Gemini model instance
        
    Raises:
        GeminiAPIError: If initialization fails
    """
    try:
        api_key = load_api_key()
        genai.configure(api_key=api_key)
        
        # Create and return the model instance
        model = genai.GenerativeModel(DEFAULT_ONLINE_MODEL)
        return model
    except Exception as e:
        if isinstance(e, GeminiAPIError):
            raise
        raise GeminiAPIError(f"Failed to initialize Gemini API: {e}")

# Define the APRS tool at the module level
get_operator_aprs_messages_func = FunctionDeclaration(
    name="get_operator_aprs_messages",
    description=(
        "Retrieves the latest received APRS messages for a specific operator's callsign. "
        "Use this when the operator asks to read their messages, check APRS mail, or similar requests. "
        "IMPORTANT: If the operator's callsign is not known from the current conversation or previous turns, "
        "you MUST ask the operator for their callsign first. Then, once they provide it, call this function again "
        "with their callsign. Do not call this function without the 'operator_callsign' parameter."
    ),
    parameters={
        "type": "object",
        "properties": {
            "operator_callsign": {
                "type": "string",
                "description": "The operator's amateur radio callsign (e.g., N0CALL, W6RGC)."
            }
        },
        "required": ["operator_callsign"]
    }
)

# Define the APRS send message tool
send_aprs_message_func = FunctionDeclaration(
    name="send_aprs_message_for_operator",
    description=(
        "Sends an APRS message from a specified sender to a recipient. "
        "Use this when the operator wants to send an APRS message. "
        "You MUST obtain the sender's callsign (usually the operator's), the recipient's callsign, "
        "and the message content (max 50 characters) before calling this function. "
        "If any of these are missing, ask the operator to provide them."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sender_callsign": {
                "type": "string",
                "description": "The callsign of the sender (e.g., W6RGC, N0CALL)."
            },
            "recipient_callsign": {
                "type": "string",
                "description": "The callsign of the recipient (e.g., N0CALL, W6RGC)."
            },
            "message_text": {
                "type": "string",
                "description": "The content of the message, 50 characters maximum."
            }
        },
        "required": ["sender_callsign", "recipient_callsign", "message_text"]
    }
)

# Define the weather tool
get_weather_forecast_func = FunctionDeclaration(
    name="get_weather_forecast",
    description=(
        "Gets current weather conditions and 3-day forecast for a specified location. "
        "Use this when the operator asks about weather, forecast, or weather conditions. "
        "Includes current temperature, conditions, humidity, and 3-day forecast with highs, lows, "
        "sky conditions, and chance of rain. If no location is specified by the operator, "
        "ask them to provide a location (city, state or city, country)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location for weather information (e.g., 'San Francisco, CA', 'New York', 'London, UK')."
            }
        },
        "required": ["location"]
    }
)

# Define the time zone tool
get_timezone_time_func = FunctionDeclaration(
    name="get_timezone_time",
    description=(
        "Gets current time and date for a specified timezone. "
        "Use this when the operator asks about time, what time it is, or time in a specific location. "
        "If no timezone is specified, defaults to Pacific Time. Supports natural language timezone inputs "
        "like 'Eastern', 'New York', 'London', 'UTC', etc. Returns time without seconds and year."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "The timezone for time information (e.g., 'Eastern', 'Pacific', 'New York', 'London', 'UTC'). Optional - defaults to Pacific Time if not specified."
            }
        },
        "required": []  # timezone is optional, defaults to Pacific
    }
)

# Create the tool with all function declarations
aprs_tool = Tool(function_declarations=[
    get_operator_aprs_messages_func, 
    send_aprs_message_func,
    get_weather_forecast_func,
    get_timezone_time_func
])

def ask_gemini(prompt: str, model_name: Optional[str] = None, 
               generation_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Send a prompt to Google Gemini and return the response.
    Handles function calls for tools like reading APRS messages.
    
    Args:
        prompt (str): The text prompt to send to Gemini
        model_name (str, optional): Gemini model to use (defaults to DEFAULT_ONLINE_MODEL)
        generation_config (dict, optional): Additional generation parameters
        
    Returns:
        str: Gemini's response text
        
    Raises:
        GeminiAPIError: If the API request fails after all retries or if function execution fails critically.
    """
    if not prompt or not prompt.strip():
        raise GeminiAPIError("Prompt cannot be empty")
    
    # Use default generation config optimized for ham radio assistant
    if generation_config is None:
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 1024,
        }
    
    # Initialize model (use custom model name if provided)
    current_model = None
    try:
        if model_name and model_name != DEFAULT_ONLINE_MODEL:
            # Ensure API key is configured for this model instance
            # genai.configure is global, load_api_key ensures it's called if needed.
            load_api_key() 
            current_model = genai.GenerativeModel(model_name)
        else:
            current_model = initialize_gemini() # Gets/initializes default model
    except Exception as e:
        # Wrap model initialization errors
        raise GeminiAPIError(f"Model initialization failed: {e}")
    
    # Attempt the API call with retries
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"🤖 Sending prompt to Gemini (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            # Generate response, including the APRS tool
            response = current_model.generate_content(
                prompt, # User's initial prompt as a string
                generation_config=generation_config,
                tools=[aprs_tool] 
            )
            
            # Check for function call in the response
            function_call_to_execute = None
            # Ensure candidate and content parts exist before accessing
            if response.candidates and \
               response.candidates[0].content and \
               response.candidates[0].content.parts:
                for part_content in response.candidates[0].content.parts:
                    if hasattr(part_content, 'function_call') and part_content.function_call:
                        function_call_to_execute = part_content.function_call
                        break # Process the first function call found
            
            if function_call_to_execute:
                fc = function_call_to_execute
                if fc.name == "get_operator_aprs_messages":
                    print(f"🛠️ Gemini requested to call function: {fc.name} with args: {fc.args}")
                    
                    operator_callsign = fc.args.get("operator_callsign")

                    if not operator_callsign:
                        # This should ideally not happen if LLM follows 'required' and description.
                        # Return a message for TTS to inform the user.
                        print("❌ Function 'get_operator_aprs_messages' called without 'operator_callsign'.")
                        return "I need your callsign to fetch APRS messages. Please tell me your callsign."

                    aprs_messages_text = ""
                    try:
                        # Execute the actual function with the operator's callsign
                        print(f"📞 Calling aprs_helper.get_aprs_messages for callsign: {operator_callsign}")
                        aprs_messages_text = get_aprs_messages(receiver=operator_callsign)
                        # TODO(richc): Add to prompt context, but ContextManager avalable here. Pass it in initialize_gemini()?
                        print(f"✉️ APRS messages received for {operator_callsign}: {aprs_messages_text[:100]}...") # Log snippet
                        if not aprs_messages_text or aprs_messages_text.strip() == "No messages found.":
                             return f"No new APRS messages found for {operator_callsign}."
                        return f"TTS_DIRECT:Messages: {aprs_messages_text}"
                        
                    except Exception as e:
                        print(f"❌ Error calling get_aprs_messages for {operator_callsign}: {e}")
                        # Return a user-friendly error message for Gemini to process or for direct TTS
                        return f"An error occurred while trying to fetch APRS messages for {operator_callsign}: {str(e)}"
                
                elif fc.name == "send_aprs_message_for_operator":
                    print(f"🛠️ Gemini requested to call function: {fc.name} with args: {fc.args}")
                    
                    sender_callsign = fc.args.get("sender_callsign")
                    recipient_callsign = fc.args.get("recipient_callsign")
                    message_text = fc.args.get("message_text")

                    if not sender_callsign or not recipient_callsign or not message_text:
                        missing_params = []
                        if not sender_callsign: missing_params.append("your callsign (sender)")
                        if not recipient_callsign: missing_params.append("recipient's callsign")
                        if not message_text: missing_params.append("message content")
                        return f"I'm missing some information to send the APRS message: {', '.join(missing_params)}. Please provide all details."

                    if len(message_text) > 50:
                        return "The APRS message is too long. Please keep it to 50 characters or less."

                    try:
                        print(f"📤 Calling aprs_helper.send_aprs_message from {sender_callsign} to {recipient_callsign}: {message_text}")
                        send_response = send_aprs_message(sender=sender_callsign, receiver=recipient_callsign, message=message_text)
                        print(f"📨 APRS send response: {send_response[:100]}...") # Log snippet
                        
                        # We need to provide a meaningful response for TTS.
                        # findu.com's sendmsg.cgi doesn't give a clear "success" or "fail" in its HTML response easily.
                        # It often just says "Message queued for delivery to..." or similar.
                        # We'll assume success if no exception, and craft a positive TTS message.
                        return f"TTS_DIRECT:APRS message sent." # from {sender_callsign} to {recipient_callsign} with message: {message_text}. The system responded: {send_response if send_response else 'OK'}."

                    except Exception as e:
                        print(f"❌ Error calling send_aprs_message from {sender_callsign} to {recipient_callsign}: {e}")
                        return f"An error occurred while trying to send the APRS message from {sender_callsign} to {recipient_callsign}: {str(e)}"
                
                elif fc.name == "get_weather_forecast":
                    print(f"🛠️ Gemini requested to call function: {fc.name} with args: {fc.args}")
                    
                    location = fc.args.get("location")

                    if not location:
                        return "I need a location to get weather information. Please tell me what city or area you'd like the weather for."

                    try:
                        print(f"🌤️ Calling weather_helper.get_weather_forecast for location: {location}")
                        weather_info = get_weather_forecast(location)
                        print(f"☀️ Weather information received for {location}: {weather_info[:100]}...") # Log snippet
                        
                        return f"TTS_DIRECT:{weather_info}"
                        
                    except Exception as e:
                        print(f"❌ Error calling get_weather_forecast for {location}: {e}")
                        return f"An error occurred while trying to get weather information for {location}: {str(e)}"
                
                elif fc.name == "get_timezone_time":
                    print(f"🛠️ Gemini requested to call function: {fc.name} with args: {fc.args}")
                    
                    timezone = fc.args.get("timezone")
                    
                    # timezone is optional - if not provided, defaults to Pacific Time
                    try:
                        if timezone:
                            print(f"🕒 Calling time_helper.get_timezone_time for timezone: {timezone}")
                        else:
                            print(f"🕒 Calling time_helper.get_timezone_time with default timezone (Pacific)")
                        
                        time_info = get_timezone_time(timezone)
                        display_tz = timezone if timezone else "default (Pacific Time)"
                        print(f"🕒 Time information received for {display_tz}: {time_info[:100]}...") # Log snippet
                        
                        return f"TTS_DIRECT:{time_info}"
                        
                    except Exception as e:
                        display_tz = timezone if timezone else "default timezone"
                        print(f"❌ Error calling get_timezone_time for {display_tz}: {e}")
                        return f"An error occurred while trying to get time information for {display_tz}: {str(e)}"
            
            # If no function call was made, or it wasn't the one we handle, proceed with normal text response
            if hasattr(response, 'text') and response.text:
                print(f"✅ Received direct response from Gemini ({len(response.text)} characters)")
                return response.text.strip()
            
            # If no text and no function call we handled, check for blocking or empty response
            block_reason_msg = ""
            if hasattr(response, 'prompt_feedback') and \
               response.prompt_feedback and \
               hasattr(response.prompt_feedback, 'block_reason') and \
               response.prompt_feedback.block_reason:
                block_reason_msg = f"Response blocked: {response.prompt_feedback.block_reason}"
                raise GeminiAPIError(block_reason_msg) # Will be caught and retried
            else:
                # This error will be caught by the outer except and retried or raised
                raise GeminiAPIError("Empty response received from Gemini (no text, no function call, or unhandled function call)")
            
        except Exception as e:
            last_error = e # Store the exception
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️  Gemini API attempt {attempt + 1} failed: {e}")
                print(f"🔄 Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else: # This is the last attempt
                print("❌ All Gemini API attempts failed.")
                # Raise the last encountered error, ensuring it's a GeminiAPIError if we wrapped it.
                if isinstance(last_error, GeminiAPIError):
                    raise last_error
                else: # Wrap other exceptions
                    raise GeminiAPIError(f"Failed to get response from Gemini after {MAX_RETRIES} attempts. Last error: {last_error}")

    # Fallback if loop finishes: should be unreachable if MAX_RETRIES >= 1 due to prior raises.
    # However, as a safeguard:
    if last_error: # This means all retries failed.
        if not isinstance(last_error, GeminiAPIError):
            last_error = GeminiAPIError(f"ask_gemini ultimately failed after retries. Last error: {last_error}")
        raise last_error
    else: # Should be truly impossible given the logic if MAX_RETRIES >=1
        raise GeminiAPIError("ask_gemini failed to produce a response or error, and no error was captured after retries.")

def ask_gemini_streaming(prompt: str, model_name: Optional[str] = None,
                        generation_config: Optional[Dict[str, Any]] = None):
    """
    Send a prompt to Google Gemini and yield response chunks as they arrive.
    Useful for real-time applications where you want to start processing
    the response before it's complete.
    
    Args:
        prompt (str): The text prompt to send to Gemini
        model_name (str, optional): Gemini model to use (defaults to DEFAULT_ONLINE_MODEL)
        generation_config (dict, optional): Additional generation parameters
        
    Yields:
        str: Response text chunks as they arrive
        
    Raises:
        GeminiAPIError: If the API request fails
    """
    if not prompt or not prompt.strip():
        raise GeminiAPIError("Prompt cannot be empty")
    
    # Use default generation config optimized for ham radio assistant
    if generation_config is None:
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 1024,
        }
    
    # Initialize model
    try:
        if model_name and model_name != DEFAULT_ONLINE_MODEL:
            api_key = load_api_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
        else:
            model = initialize_gemini()
    except Exception as e:
        raise GeminiAPIError(f"Model initialization failed: {e}")
    
    try:
        print(f"🤖 Starting streaming response from Gemini...")
        
        # Generate streaming response
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        raise GeminiAPIError(f"Streaming request failed: {e}")

def test_gemini_connection() -> bool:
    """
    Test the Gemini API connection with a simple prompt.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        test_response = ask_gemini("Hello, please respond with 'Connection successful'")
        return "successful" in test_response.lower()
    except Exception as e:
        print(f"❌ Gemini connection test failed: {e}")
        return False

def list_available_models() -> list:
    """
    List all available Gemini models.
    
    Returns:
        list: List of available model names
    """
    try:
        api_key = load_api_key()
        genai.configure(api_key=api_key)
        
        models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models.append(model.name)
        
        return models
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Gemini API integration...")
    
    try:
        # First, list available models
        print("📋 Listing available Gemini models...")
        available_models = list_available_models()
        if available_models:
            print("✅ Available models:")
            for model in available_models:
                print(f"   - {model}")
            print(f"\n🎯 Using model: {DEFAULT_ONLINE_MODEL}")
        else:
            print("⚠️  Could not retrieve model list")
        
        # Test basic connection
        if test_gemini_connection():
            print("✅ Gemini API connection successful!")
            
            # Test a ham radio related query
            test_prompt = "What is the difference between FM and SSB in ham radio?"
            response = ask_gemini(test_prompt)
            print(f"\n📝 Test Query: {test_prompt}")
            print(f"🤖 Gemini Response: {response}")
            
            # Test streaming (just show first few chunks)
            print(f"\n🔄 Testing streaming response...")
            chunk_count = 0
            for chunk in ask_gemini_streaming("Explain how a dipole antenna works"):
                print(f"Chunk {chunk_count + 1}: {chunk[:50]}...")
                chunk_count += 1
                if chunk_count >= 3:  # Just show first 3 chunks
                    break
            
        else:
            print("❌ Gemini API connection failed!")
            
    except GeminiAPIError as e:
        print(f"❌ Gemini API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 