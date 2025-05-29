#!/usr/bin/env python3
# main.py - Off Grid Ham Radio AI Voice Assistant
#
# This is the main entry point for the W6RGC/AI Ham Radio Voice Assistant.
# It integrates wake word detection, speech recognition, AI conversation,
# and text-to-speech for a complete voice assistant experience over ham radio.
#
# Key Features:
#  - AST-based wake word detection: "seven" (from 35+ available options)
#  - Whisper speech recognition for accurate transcription
#  - Ollama LLM integration for conversational AI
#  - CoquiTTS for natural-sounding speech synthesis
#  - AIOC adapter support for PTT control
#  - Automatic audio device detection and configuration
#  - Centralized configuration through constants.py
#
# Hardware Requirements:
#  - AIOC (All-In-One-Cable) adapter or compatible USB audio interface
#  - Serial port for PTT control (typically /dev/ttyACM0)
#  - Microphone and speakers/headphones
#  - Optional: CUDA-capable GPU for acceleration
#
# Usage:
#  1. Say the wake word "seven" (or configured alternative)
#  2. Speak your command or question
#  3. The AI will respond via TTS and transmit over radio
#  4. Say "seven, break" or "seven, exit" to shutdown
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
# This app integrates multiple AI technologies for ham radio operations:
#  - Dual wake word detection methods:
#    * AST Method: MIT/ast-finetuned-speech-commands-v2 (35+ wake words, efficient)
#    * Custom Method: Whisper + energy detection (flexible, any phrase)
#  - OpenAI Whisper voice recognition engine for high-quality speech-to-text
#  - Ollama LLM engine as the AI chatbot (currently gemma3:12b)
#  - CoquiTTS text-to-speech engine with CUDA acceleration
#  - Automatic AIOC (All-In-One-Cable) adapter detection and configuration
#  - Serial PTT control for radio transmission
#  - Sample rate conversion (48kHz device → 16kHz model requirements)
#
# Current Configuration:
#  - AST wake word: "seven" (from 35+ available options)
#  - Bot name: "7" (configurable in context_manager.py)
#  - Audio device: Auto-detected AIOC adapter
#  - Serial port: Auto-detected or manually set to /dev/ttyACM0 or /dev/ttyACM1

import sounddevice as sd
import numpy as np
import soundfile as sf
import whisper
from scipy.signal import resample
import time
import requests
import os
import json
from TTS.api import TTS as CoquiTTS
import re
import wake_word_detector

import commands # Import the new commands module
from context_manager import ContextManager
from ril_aioc import RadioInterfaceLayerAIOC
from ril_digirig import RadioInterfaceLayerDigiRig # New import for Digirig
from periodically_identify import PeriodicIdentifier # New import for periodic ID
from llm_ollama_offline import ask_ollama # Import LLM function

### CONSTANTS ###

from constants import (
    # Bot name and phonetic call sign
    BOT_NAME,
    BOT_PHONETIC_CALLSIGN,

    # Audio processing constants
    AUDIO_THRESHOLD,
    SILENCE_DURATION,
    FRAME_DURATION,
    WHISPER_TARGET_SAMPLE_RATE,
    
    # AI/LLM configuration
    OLLAMA_URL,
    DEFAULT_MODEL,
    
    # TTS configuration
    TTS_MODEL_FAST_PITCH,
    TTS_MODEL_SPEEDY_SPEECH,
    TTS_MODEL_TACOTRON2,
    TTS_INFERENCE_NOISE_SCALE,
    TTS_INFERENCE_NOISE_SCALE_DP,
    TTS_INFERENCE_SIGMA,
    TTS_OUTPUT_FILE,
    
    # Hardware configuration
    DEFAULT_AIOC_SERIAL_PORT,
    DEFAULT_DIGIRIG_SERIAL_PORT,
    RIL_TYPE_AIOC,
    RIL_TYPE_DIGIRIG,
    DEFAULT_RIL_TYPE,
    
    # Wake word detection
    WAKE_WORD_METHOD_AST,
    AST_MODEL_NAME,
    DEFAULT_WAKE_WORD
)

### HELPER FUNCTIONS ###

def create_radio_interface_layer(ril_type=DEFAULT_RIL_TYPE, serial_port_name=None):
    """
    Factory function to create the appropriate Radio Interface Layer instance.
    
    Args:
        ril_type (str): Type of RIL to create ("aioc" or "digirig")
        serial_port_name (str): Serial port for PTT control (auto-selected if None)
        
    Returns:
        RadioInterfaceLayer instance (AIOC or Digirig)
        
    Raises:
        ValueError: If ril_type is not supported
        RuntimeError: If RIL initialization fails
    """
    if ril_type.lower() == RIL_TYPE_AIOC:
        if serial_port_name is None:
            serial_port_name = DEFAULT_AIOC_SERIAL_PORT
        print(f"🔧 Initializing AIOC Radio Interface Layer...")
        return RadioInterfaceLayerAIOC(serial_port_name=serial_port_name)
    elif ril_type.lower() == RIL_TYPE_DIGIRIG:
        if serial_port_name is None:
            serial_port_name = DEFAULT_DIGIRIG_SERIAL_PORT
        print(f"🔧 Initializing Digirig Radio Interface Layer...")
        return RadioInterfaceLayerDigiRig(serial_port_name=serial_port_name)
    else:
        raise ValueError(f"Unsupported RIL type: {ril_type}. Supported types: {RIL_TYPE_AIOC}, {RIL_TYPE_DIGIRIG}")



def play_tts_audio_fast(text_to_speak, tts_engine, aioc_interface):
    """
    Ultra-fast TTS audio playback using in-memory processing (no file I/O).
    Handles PTT, audio preparation, and uses RadioInterfaceLayerAIOC for sd.play().
    """
    try:
        aioc_interface.reset_audio_device() # Reset before generating/playing audio
        
        print("🔊 Generating TTS audio (in-memory)...")
        tts_audio_data = tts_engine.tts(text=text_to_speak)
        tts_sample_rate = tts_engine.synthesizer.output_sample_rate
        
        if not isinstance(tts_audio_data, np.ndarray):
            tts_audio_data = np.array(tts_audio_data, dtype=np.float32)
        
        if tts_audio_data.ndim > 1 and tts_audio_data.shape[1] > 1:
            tts_audio_data = np.mean(tts_audio_data, axis=1)
        
        max_val = np.max(np.abs(tts_audio_data))
        if max_val > 0:
            tts_audio_data = tts_audio_data * 0.95 / max_val
        
        print(f"🔊 Prepared audio for RIL ({len(tts_audio_data)/tts_sample_rate:.1f}s). PTT ON.")
        aioc_interface.ptt_on()
        time.sleep(0.2) # PTT engage delay
        aioc_interface.play_audio(tts_audio_data, tts_sample_rate)
        
    except Exception as e:
        print(f"❌ Error in fast TTS playback (main.py): {e}")
        print("🔄 Falling back to file-based method...")
        play_tts_audio(text_to_speak, tts_engine, aioc_interface) # Fallback already handles PTT
    finally:
        aioc_interface.ptt_off() # Ensure PTT is off
        # Force garbage collection to prevent memory accumulation
        import gc
        if 'tts_audio_data' in locals():
            del tts_audio_data
        gc.collect()
        print("✅ Audio transmission cycle complete (fast mode). PTT OFF.")

def play_tts_audio(text_to_speak, tts_engine, aioc_interface):
    """
    Optimized TTS audio playback with file-based fallback.
    Handles PTT, audio preparation, and uses RadioInterfaceLayerAIOC for sd.play().
    """
    try:
        aioc_interface.reset_audio_device() # Reset before generating/playing audio

        print("🔊 Generating TTS audio (file-based fallback)...")
        tts_engine.tts_to_file(
            text=text_to_speak, 
            file_path=TTS_OUTPUT_FILE,
        )
        tts_audio_data, tts_sample_rate = sf.read(TTS_OUTPUT_FILE, dtype='float32')
        
        if tts_audio_data.ndim > 1 and tts_audio_data.shape[1] > 1:
            tts_audio_data = np.mean(tts_audio_data, axis=1)
        
        max_val = np.max(np.abs(tts_audio_data))
        if max_val > 0:
            tts_audio_data = tts_audio_data * 0.95 / max_val
        
        print(f"🔊 Prepared audio for RIL ({len(tts_audio_data)/tts_sample_rate:.1f}s). PTT ON.")
        aioc_interface.ptt_on()
        time.sleep(0.3) # PTT engage delay (slightly longer for file method just in case)
        aioc_interface.play_audio(tts_audio_data, tts_sample_rate)

    except Exception as e:
        print(f"❌ Error in TTS playback (main.py file-based): {e}")
    finally:
        aioc_interface.ptt_off() # Ensure PTT is off
        if os.path.exists(TTS_OUTPUT_FILE):
            os.remove(TTS_OUTPUT_FILE)
        import gc
        if 'tts_audio_data' in locals():
            del tts_audio_data
        gc.collect()
        print("✅ Audio transmission cycle complete (file-based). PTT OFF.")

def get_full_command_after_wake_word(aioc_interface, model):
    """
    Record and transcribe the full command after wake word detection.
    Uses the existing Whisper model for high-quality transcription.
    Uses RadioInterfaceLayerAIOC for audio input stream parameters.
    """
    print("🎤 Recording your command...")
    
    # Reset audio device to ensure clean recording
    aioc_interface.reset_audio_device()
    
    recording = []
    speech_started = False
    silence_counter = 0
    
    # Get stream parameters from RIL
    stream_params = aioc_interface.get_input_stream_params()
    samplerate = stream_params['samplerate'] # Use RIL determined samplerate

    with sd.InputStream(**stream_params) as stream:
        while True:
            frame, _ = stream.read(int(FRAME_DURATION * samplerate))
            frame = np.squeeze(frame)
            amplitude = np.max(np.abs(frame))
            
            if amplitude > AUDIO_THRESHOLD:
                if not speech_started:
                    print("🗣️  Speech detected, recording...")
                    speech_started = True
                recording.append(frame)
                silence_counter = 0
            elif speech_started:
                recording.append(frame)
                silence_counter += FRAME_DURATION
                if silence_counter >= SILENCE_DURATION:
                    print("🔇 Silence detected, processing...")
                    break
    
    if not recording:
        print("❌ No speech detected")
        return ""
    
    # Process audio for Whisper
    audio = np.concatenate(recording)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if samplerate != WHISPER_TARGET_SAMPLE_RATE:
        num_samples = int(len(audio) * WHISPER_TARGET_SAMPLE_RATE / samplerate)
        audio = resample(audio, num_samples)
    
    # Transcribe with main Whisper model
    print("📝 Transcribing with Whisper...")
    result = model.transcribe(audio, fp16=True, language='en')
    transcribed_text = result['text'].strip()
    
    print(f"📝 Transcribed: '{transcribed_text}'")
    return transcribed_text

### INITIALIZATION ###

## Radio Interface Layer setup
# Initialize the appropriate Radio Interface Layer based on configuration
try:
    print(f"🔧 Selected RIL type: {DEFAULT_RIL_TYPE.upper()}")
    ril = create_radio_interface_layer(ril_type=DEFAULT_RIL_TYPE)
    
    # Get necessary info from the RIL instance (same interface for both AIOC and Digirig)
    audio_index = ril.get_audio_device_index()
    samplerate = ril.get_samplerate()
    channels = ril.get_channels()
    # serial_conn is now managed by the RIL instance
except ValueError as e:
    print(f"[CONFIGURATION ERROR] {e}")
    print(f"Please check the DEFAULT_RIL_TYPE setting in constants.py")
    exit()
except RuntimeError as e:
    print(f"[CRITICAL ERROR] Could not initialize {DEFAULT_RIL_TYPE.upper()} Radio Interface Layer: {e}")
    print("The application cannot continue without the radio interface. Please check connections and configurations.")
    exit()
except Exception as e:
    print(f"[CRITICAL ERROR] An unexpected error occurred during RIL initialization: {e}")
    exit()

# Initialize ContextManager
context_mgr = ContextManager() # Initialize the context manager

# Initialize Whisper voice recognition
model = whisper.load_model("small")

# Initialize CoquiTTS with a faster model for better real-time performance
# Using FastSpeech2 which is much faster than Tacotron2 for real-time applications
try:
    # Try FastSpeech2 first (fastest)
    coqui_tts_engine = CoquiTTS(model_name=TTS_MODEL_FAST_PITCH, progress_bar=True, gpu=True)
    print("✅ Using FastPitch TTS model (optimized for speed)")
except:
    try:
        # Fallback to a simpler, faster model
        coqui_tts_engine = CoquiTTS(model_name=TTS_MODEL_SPEEDY_SPEECH, progress_bar=True, gpu=True)
        print("✅ Using SpeedySpeech TTS model (fast)")
    except:
        # Final fallback to original but with speed optimizations
        coqui_tts_engine = CoquiTTS(model_name=TTS_MODEL_TACOTRON2, progress_bar=True, gpu=True)
        print("⚠️  Using Tacotron2-DDC (slower but reliable)")
        
# Configure TTS for speed over quality
if hasattr(coqui_tts_engine, 'synthesizer') and hasattr(coqui_tts_engine.synthesizer, 'tts_config'):
    # Try to set faster inference settings
    try:
        coqui_tts_engine.synthesizer.tts_config.inference_noise_scale = TTS_INFERENCE_NOISE_SCALE
        coqui_tts_engine.synthesizer.tts_config.inference_noise_scale_dp = TTS_INFERENCE_NOISE_SCALE_DP
        coqui_tts_engine.synthesizer.tts_config.inference_sigma = TTS_INFERENCE_SIGMA
        print("✅ TTS speed optimizations applied")
    except:
        print("⚠️  Could not apply TTS speed optimizations")

# Initialize wake word detector using AST method
wake_detector = wake_word_detector.create_wake_word_detector(
    method=WAKE_WORD_METHOD_AST, 
    device_sample_rate=samplerate,
    wake_word=DEFAULT_WAKE_WORD
)

# Initialize periodic identifier
periodic_identifier = PeriodicIdentifier(
    tts_engine=coqui_tts_engine,
    aioc_interface=ril,
    play_tts_function=play_tts_audio  # Use the file-based method for reliability
)

print("🚀 Ham radio AI voice assistant starting up...")
print(f"Wake word detector: Ready (AST method, wake word: '{DEFAULT_WAKE_WORD}')")
print(f"Speech recognition: Whisper version:{model._version}")
print(f"AI model: {DEFAULT_MODEL}")
print(f"Text-to-speech: {coqui_tts_engine.model_name}")
print("=" * 50)

### MAIN LOOP ###

# Start periodic identification
periodic_identifier.start()

while True:
    try:
        # Step 1: Wait for wake word detection
        print(f"\n🎤 Listening for wake word '{DEFAULT_WAKE_WORD}'...")
        
        # AST detector returns boolean
        wake_detected = wake_detector.listen_for_wake_word(
            audio_device_index=audio_index,
            debug=False
        )
        
        if not wake_detected:
            print("❌ Wake word not detected, continuing to listen...")
            continue
            
        print(f"✅ Wake word '{DEFAULT_WAKE_WORD}' detected! Now listening for your command...")
        operator_text = get_full_command_after_wake_word(ril, model)
        
        # Step 2: Process the command

        # RICHC: This is a hack to get the wake word detector to pass the name of the bot. 
        operator_text = f"{BOT_NAME}, {operator_text}"
        # Assumes the wake word is the same as the bots name.
        print(f"🗣️  Processing command: '{operator_text}'")
        
        # Identify the command using the commands module
        command_type = commands.handle_command(operator_text)

        if command_type == "terminate":
            print("🛑 Termination command identified by main.py. Shutting down...")
            play_tts_audio(f"Terminating. Have a nice day! This is {BOT_PHONETIC_CALLSIGN} shutting down my " +
                           "processes. I am clear. Seven three.", coqui_tts_engine, ril)
            break
        elif command_type == "status":
            print("⚙️ Status command identified by main.py.")
            status_report = f"""I am {BOT_NAME}. All systems are go. I use:
                the {DEFAULT_MODEL} large language model for intelligence, 
                the {AST_MODEL_NAME} for wake word detection, 
                the Whisper version {model._version} for speech recognition, and 
                the {coqui_tts_engine.model_name} for text-to-speech."""
            play_tts_audio(status_report, coqui_tts_engine, ril)
            continue
        elif command_type == "reset":
            print("🔄 Reset command identified by main.py. Resetting context.")
            context_mgr.reset_context()
            play_tts_audio("My context has been reset. I am ready for a new conversation.", 
                           coqui_tts_engine, ril)
            continue
        elif command_type == "identify":
            print("🆔 Identify command identified by main.py.")
            identify_response = f"This is {BOT_PHONETIC_CALLSIGN}."
            play_tts_audio(identify_response, coqui_tts_engine, ril)
            periodic_identifier.restart_timer
            continue
        else:
            # If no command was handled, proceed with conversation
            print(f"💬 Conversation request detected")
            
            # Ask AI: Send transcribed test from the operator to the AI
            current_prompt = context_mgr.add_operator_request_to_context(operator_text)
            print("🤖 Sending to Ollama...")
            print(f"Current prompt: {current_prompt}")
            ollama_response = ask_ollama(current_prompt)
            print(f"🤖 Ollama replied: {ollama_response}")
            context_mgr.add_ai_response_to_context(ollama_response)
            
            # Speak response
            print("🔊 Speaking response...")
            ril.reset_audio_device() # Reset audio device before TTS to prevent conflicts
            play_tts_audio_fast(ollama_response, coqui_tts_engine, ril)
            periodic_identifier.restart_timer

        # Reset audio device after TTS for next recording cycle
        ril.reset_audio_device()            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Shutting down...")
        break
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
        continue

print("🏁 Ham Radio AI Assistant shutdown complete.")
periodic_identifier.stop() # Stop periodic identification
ril.close() # Close the RIL (which closes serial conn)