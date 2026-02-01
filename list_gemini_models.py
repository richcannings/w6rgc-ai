#!/usr/bin/env python3
# list_gemini_models.py - List Available Google Gemini Models
#
# This script lists all available Gemini models that can be used with the
# W6RGC/AI Ham Radio Voice Assistant. It shows model names, supported features,
# and other relevant information.
#
# Usage:
#   python list_gemini_models.py
#
# Requirements:
#  - google-genai library (pip install google-genai)
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
try:
    from google import genai
except ImportError as exc:
    raise ImportError(
        "google-genai is required. Install it with `pip install google-genai`."
    ) from exc
from constants import (
    GEMINI_API_KEY_FILE,
    DEFAULT_ONLINE_MODEL
)

def load_api_key():
    """Load the Gemini API key from the gemini_api_key.txt file."""
    try:
        if not os.path.exists(GEMINI_API_KEY_FILE):
            print(f"❌ Error: API key file '{GEMINI_API_KEY_FILE}' not found")
            print(f"Please create this file with your Google AI API key")
            return None
        
        with open(GEMINI_API_KEY_FILE, 'r') as f:
            api_key = f.read().strip()
        
        if not api_key:
            print(f"❌ Error: API key file '{GEMINI_API_KEY_FILE}' is empty")
            return None
        
        return api_key
    except Exception as e:
        print(f"❌ Error reading API key file: {e}")
        return None

def list_gemini_models():
    """List all available Gemini models with details."""
    
    # Load and configure API key
    api_key = load_api_key()
    if not api_key:
        return False
    
    try:
        client = genai.Client(api_key=api_key)
        print("🔗 Connected to Google AI API successfully!")
        print("=" * 60)
        
        # Get list of models
        models = list(client.models.list())
        
        if not models:
            print("⚠️  No models found")
            return False
        
        print(f"📋 Found {len(models)} total models:")
        print()
        
        # Separate models by capability
        text_generation_models = []
        other_models = []
        
        for model in models:
            supported_actions = getattr(model, 'supported_actions', None) or \
                getattr(model, 'supported_generation_methods', None) or []
            if 'generateContent' in supported_actions:
                text_generation_models.append(model)
            else:
                other_models.append(model)
        
        # Display text generation models (what we care about)
        print("🤖 TEXT GENERATION MODELS (Compatible with W6RGC/AI):")
        print("=" * 60)
        
        if text_generation_models:
            for i, model in enumerate(text_generation_models, 1):
                name = model.name
                display_name = getattr(model, 'display_name', 'N/A')
                description = getattr(model, 'description', 'No description available')
                
                # Highlight the current default model
                if DEFAULT_ONLINE_MODEL in name:
                    print(f"✅ {i:2}. {name} ⭐ (CURRENT DEFAULT)")
                else:
                    print(f"   {i:2}. {name}")
                
                if display_name != 'N/A':
                    print(f"       Display Name: {display_name}")
                
                print(f"       Description: {description}")
                print(f"       Supported Methods: {', '.join(supported_actions)}")
                
                # Show input/output token limits if available
                if hasattr(model, 'input_token_limit'):
                    print(f"       Input Token Limit: {model.input_token_limit:,}")
                if hasattr(model, 'output_token_limit'):
                    print(f"       Output Token Limit: {model.output_token_limit:,}")
                
                print()
        else:
            print("   No text generation models found")
        
        # Display other models for completeness
        if other_models:
            print("🔧 OTHER MODELS (Not compatible with text generation):")
            print("=" * 60)
            for i, model in enumerate(other_models, 1):
                print(f"   {i:2}. {model.name}")
                other_supported_actions = getattr(model, 'supported_actions', None) or \
                    getattr(model, 'supported_generation_methods', None) or []
                print(f"       Methods: {', '.join(other_supported_actions)}")
                print()
        
        # Summary
        print("📊 SUMMARY:")
        print("=" * 60)
        print(f"Total models: {len(models)}")
        print(f"Text generation models: {len(text_generation_models)}")
        print(f"Current default model: {DEFAULT_ONLINE_MODEL}")
        print()
        print("💡 To use a different model, update DEFAULT_ONLINE_MODEL in constants.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

def test_model(model_name):
    """Test a specific model with a simple query."""
    try:
        api_key = load_api_key()
        if not api_key:
            return False
            
        client = genai.Client(api_key=api_key)
        
        print(f"🧪 Testing model: {model_name}")
        test_prompt = "Hello! Please respond with 'Test successful' if you can understand this message."
        
        response = client.models.generate_content(model=model_name, contents=test_prompt)
        
        if response.text:
            print(f"✅ Test Response: {response.text.strip()}")
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function to run the model listing script."""
    print("🚀 Gemini Models List for W6RGC/AI")
    print("=" * 60)
    print()
    
    # List all models
    success = list_gemini_models()
    
    if success:
        print()
        # Ask if user wants to test the current default model
        try:
            test_default = input(f"🧪 Test current default model ({DEFAULT_ONLINE_MODEL})? (y/N): ").strip().lower()
            if test_default in ['y', 'yes']:
                print()
                test_model(DEFAULT_ONLINE_MODEL)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except EOFError:
            pass  # Handle when running without interactive terminal
    
    print("\n✨ Script complete!")

if __name__ == "__main__":
    main() 