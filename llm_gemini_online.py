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
from typing import Optional, Dict, Any
import time

# Configuration constants
GEMINI_API_KEY_FILE = "gemini_api_key.txt"
DEFAULT_MODEL = "gemini-1.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
REQUEST_TIMEOUT = 30  # seconds

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
        model = genai.GenerativeModel(DEFAULT_MODEL)
        return model
    except Exception as e:
        if isinstance(e, GeminiAPIError):
            raise
        raise GeminiAPIError(f"Failed to initialize Gemini API: {e}")

def ask_gemini(prompt: str, model_name: Optional[str] = None, 
               generation_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Send a prompt to Google Gemini and return the response.
    
    Args:
        prompt (str): The text prompt to send to Gemini
        model_name (str, optional): Gemini model to use (defaults to DEFAULT_MODEL)
        generation_config (dict, optional): Additional generation parameters
        
    Returns:
        str: Gemini's response text
        
    Raises:
        GeminiAPIError: If the API request fails after all retries
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
    try:
        if model_name and model_name != DEFAULT_MODEL:
            api_key = load_api_key()
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
        else:
            model = initialize_gemini()
    except Exception as e:
        raise GeminiAPIError(f"Model initialization failed: {e}")
    
    # Attempt the API call with retries
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"🤖 Sending prompt to Gemini (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Check if response was blocked or empty
            if not response.text:
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if hasattr(feedback, 'block_reason'):
                        raise GeminiAPIError(f"Response blocked: {feedback.block_reason}")
                raise GeminiAPIError("Empty response received from Gemini")
            
            print(f"✅ Received response from Gemini ({len(response.text)} characters)")
            return response.text.strip()
            
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️  Gemini API attempt {attempt + 1} failed: {e}")
                print(f"🔄 Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"❌ All Gemini API attempts failed")
    
    # If we get here, all retries failed
    raise GeminiAPIError(f"Failed to get response from Gemini after {MAX_RETRIES} attempts. Last error: {last_error}")

def ask_gemini_streaming(prompt: str, model_name: Optional[str] = None,
                        generation_config: Optional[Dict[str, Any]] = None):
    """
    Send a prompt to Google Gemini and yield response chunks as they arrive.
    Useful for real-time applications where you want to start processing
    the response before it's complete.
    
    Args:
        prompt (str): The text prompt to send to Gemini
        model_name (str, optional): Gemini model to use (defaults to DEFAULT_MODEL)
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
        if model_name and model_name != DEFAULT_MODEL:
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
            print(f"\n🎯 Using model: {DEFAULT_MODEL}")
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