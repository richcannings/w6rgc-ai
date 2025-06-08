#!/usr/bin/env python3
# test_all_features.py - Test all features together

from llm_gemini_online import ask_gemini

def test_all_features():
    """Test all features to ensure they work together."""
    print("🧪 Testing All Features Together...")
    
    try:
        print("\n🕒 1. Time Feature:")
        response = ask_gemini("What time is it?")
        print(f"   Response: {response}")
        
        print("\n🌤️ 2. Weather Feature:")  
        response = ask_gemini("What is the weather in New York?")
        print(f"   Response: {response}")
        
        print("\n🌍 3. Time in Different Timezone:")
        response = ask_gemini("What time is it in London?")
        print(f"   Response: {response}")
        
        print("\n📝 4. Combined Question (if Gemini can handle it):")
        response = ask_gemini("What time is it in Tokyo and what is the weather there?")
        print(f"   Response: {response}")
        
        print("\n✅ All features test completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_all_features() 