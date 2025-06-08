#!/usr/bin/env python3
# test_weather_integration.py - Test weather integration with Gemini

from llm_gemini_online import ask_gemini

def test_weather_integration():
    """Test the weather integration with Gemini."""
    print("🧪 Testing Weather Integration with Gemini...")
    
    # Test weather request
    try:
        print("\n📝 Testing weather request...")
        response = ask_gemini("What's the weather like in San Francisco, CA?")
        print(f"🤖 Gemini Response: {response}")
        
        print("\n📝 Testing forecast request...")
        response = ask_gemini("Can you give me the 3-day weather forecast for New York?")
        print(f"🤖 Gemini Response: {response}")
        
        print("\n📝 Testing weather request without location...")
        response = ask_gemini("What's the weather like?")
        print(f"🤖 Gemini Response: {response}")
        
    except Exception as e:
        print(f"❌ Error testing weather integration: {e}")

if __name__ == "__main__":
    test_weather_integration() 