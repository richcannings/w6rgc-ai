#!/usr/bin/env python3
"""
TTS Performance Test Script
Tests different TTS methods to identify performance bottlenecks.
"""

import time
import numpy as np
from TTS.api import TTS
import sounddevice as sd
from constants import (
    TEST_TTS_TEXT,
    TTS_MODEL_FAST_PITCH,
    TTS_MODEL_SPEEDY_SPEECH,
    TTS_MODEL_TACOTRON2,
    TEST_DURATION,
    DEFAULT_DEVICE_SAMPLE_RATE,
    TEST_FREQUENCY,
    TEST_AMPLITUDE
)

def test_tts_performance():
    """Test TTS performance with different methods."""
    
    print("🧪 TTS Performance Test")
    print("=" * 50)
    
    # Initialize TTS with different models
    models_to_test = [
        TTS_MODEL_FAST_PITCH,
        TTS_MODEL_SPEEDY_SPEECH, 
        TTS_MODEL_TACOTRON2
    ]
    
    for model_name in models_to_test:
        print(f"\n🔬 Testing model: {model_name}")
        
        try:
            # Initialize TTS
            start_time = time.time()
            tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
            init_time = time.time() - start_time
            print(f"   ⏱️  Model loading: {init_time:.2f}s")
            
            # Test in-memory generation
            start_time = time.time()
            audio_data = tts.tts(text=TEST_TTS_TEXT)
            generation_time = time.time() - start_time
            
            # Get sample rate
            sample_rate = tts.synthesizer.output_sample_rate
            audio_duration = len(audio_data) / sample_rate
            
            print(f"   ⏱️  Audio generation: {generation_time:.2f}s")
            print(f"   🎵 Audio duration: {audio_duration:.2f}s")
            print(f"   ⚡ Real-time factor: {generation_time/audio_duration:.2f}x")
            
            # Test file-based generation
            start_time = time.time()
            tts.tts_to_file(text=TEST_TTS_TEXT, file_path="test_output.wav")
            file_time = time.time() - start_time
            print(f"   ⏱️  File generation: {file_time:.2f}s")
            print(f"   📈 File vs Memory: {file_time/generation_time:.2f}x slower")
            
            # Clean up
            import os
            if os.path.exists("test_output.wav"):
                os.remove("test_output.wav")
                
        except Exception as e:
            print(f"   ❌ Error testing {model_name}: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Recommendations:")
    print("   • Use in-memory generation (tts.tts()) instead of file I/O")
    print("   • Choose fastest model with acceptable quality")
    print("   • Real-time factor < 1.0 is ideal for real-time applications")

def test_audio_device_performance():
    """Test audio device performance."""
    print("\n🔊 Audio Device Performance Test")
    print("=" * 30)
    
    # Generate test tone
    duration = TEST_DURATION  # seconds
    sample_rate = DEFAULT_DEVICE_SAMPLE_RATE
    frequency = TEST_FREQUENCY  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = TEST_AMPLITUDE * np.sin(2 * np.pi * frequency * t)
    
    # Test different buffer sizes
    for blocksize in [512, 1024, 2048, 4096]:
        print(f"   Testing blocksize: {blocksize}")
        
        try:
            start_time = time.time()
            sd.play(test_audio, sample_rate, blocksize=blocksize, blocking=True)
            play_time = time.time() - start_time
            
            print(f"   ⏱️  Playback time: {play_time:.2f}s (expected: {duration:.2f}s)")
            
            if abs(play_time - duration) < 0.1:
                print(f"   ✅ Good performance with blocksize {blocksize}")
            else:
                print(f"   ⚠️  Timing issues with blocksize {blocksize}")
                
        except Exception as e:
            print(f"   ❌ Error with blocksize {blocksize}: {e}")

if __name__ == "__main__":
    test_tts_performance()
    test_audio_device_performance()
    print("\n🏁 Performance testing complete!") 