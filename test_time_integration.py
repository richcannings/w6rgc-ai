#!/usr/bin/env python3
# test_time_integration.py - Test time integration with Gemini

from llm_gemini_online import ask_gemini

def test_time_integration():
    """Test the time integration with Gemini."""
    print("🧪 Testing Time Integration with Gemini...")
    
    # Test time requests
    try:
        print("\n📝 Testing default time request...")
        response = ask_gemini("What time is it?")
        print(f"🤖 Gemini Response: {response}")
        
        print("\n📝 Testing Eastern time request...")
        response = ask_gemini("What time is it in Eastern time?")
        print(f"🤖 Gemini Response: {response}")
        
        print("\n📝 Testing specific city time request...")
        response = ask_gemini("What time is it in London?")
        print(f"🤖 Gemini Response: {response}")
        
        print("\n📝 Testing UTC time request...")
        response = ask_gemini("What's the current UTC time?")
        print(f"🤖 Gemini Response: {response}")
        
        print("\n📝 Testing time in a different city...")
        response = ask_gemini("Can you tell me what time it is in Tokyo?")
        print(f"🤖 Gemini Response: {response}")
        
    except Exception as e:
        print(f"❌ Error testing time integration: {e}")

if __name__ == "__main__":
    test_time_integration() 