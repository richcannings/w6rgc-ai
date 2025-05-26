#!/usr/bin/env python3
# main.py - an AI voice assistant for ham radio
#
# Author: Rich Cannings, W6RGC, rcannings@gmail.com
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
#  - AST wake word: "seven" (recommended for efficiency)
#  - Custom wake word: "Overlord" (for flexibility)
#  - Bot name: "7" (configurable in prompts.py)
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
import serial
import json
from TTS.api import TTS as CoquiTTS
import re
import wake_word_detector

import prompts
from prompts import BOT_NAME 
from prompts import BOT_PHONETIC_CALLSIGN

### CONSTANTS ###

# Parameters for voice activity detection
THRESHOLD = 0.02  # Adjust as needed for your mic/environment
SILENCE_DURATION = 2.0  # seconds of silence to consider end of speech, originally 1.0
FRAME_DURATION = 0.1  # seconds per audio frame

# Parameters for using Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b" #"gemma3:4b"  # Change to your preferred model

### HELPER FUNCTIONS ###

def find_aioc_device_index():
    """
    Find the audio device index for the AIOC (All-In-One-Cable) adapter.
    Returns the device index if found, otherwise raises an exception.
    """
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if "All-In-One-Cable" in device['name']:
            # Verify it has both input and output capabilities
            if device['max_input_channels'] > 0 and device['max_output_channels'] > 0:
                print(f"Found AIOC device at index {i}: {device['name']}")
                return i
    
    # If not found, list available devices to help with debugging
    print("AIOC device not found. Available audio devices:")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device['name']} (in:{device['max_input_channels']}, out:{device['max_output_channels']})")
    
    raise RuntimeError("AIOC (All-In-One-Cable) device not found. Please check your USB connection.")

def convert_ollama_response(response_text):
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
    payload = {
        "model": MODEL,
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

def play_tts_audio(text_to_speak, tts_engine, serial_conn, audio_device_index):
    tts_wav_file = 'ollama_tts.wav'
    tts_engine.tts_to_file(text=text_to_speak, file_path=tts_wav_file) # TODO: can I save at a faster, lower quality?
    tts_audio_data, tts_sample_rate = sf.read(tts_wav_file, dtype='float32')
    serial_conn.setRTS(True)
    serial_conn.setDTR(True)
    time.sleep(0.5)
    sd.play(tts_audio_data, tts_sample_rate, device=audio_device_index)
    sd.wait()
    serial_conn.setRTS(False)
    serial_conn.setDTR(False)
    os.remove(tts_wav_file)

def get_full_command_after_wake_word(audio_index, samplerate, channels, model):
    """
    Record and transcribe the full command after wake word detection.
    Uses the existing Whisper model for high-quality transcription.
    """
    print("🎤 Recording your command...")
    
    recording = []
    speech_started = False
    silence_counter = 0
    
    # Use same parameters as before
    FRAME_DURATION = 0.1
    SILENCE_DURATION = 2.0
    THRESHOLD = 0.02
    
    with sd.InputStream(samplerate=samplerate, channels=channels, device=audio_index, dtype='float32') as stream:
        while True:
            frame, _ = stream.read(int(FRAME_DURATION * samplerate))
            frame = np.squeeze(frame)
            amplitude = np.max(np.abs(frame))
            
            if amplitude > THRESHOLD:
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
    if samplerate != 16000:
        num_samples = int(len(audio) * 16000 / samplerate)
        audio = resample(audio, num_samples)
    
    # Transcribe with main Whisper model
    print("📝 Transcribing with Whisper...")
    result = model.transcribe(audio, fp16=True, language='en')
    transcribed_text = result['text'].strip()
    
    print(f"📝 Transcribed: '{transcribed_text}'")
    return transcribed_text

### INITIALIZATION ###

## AIOC adapter setup
# audio_index will be determined automatically by find_aioc_device_index()
# Find and set up microphone and output device for AIOC adapter
audio_index = find_aioc_device_index()
device_info = sd.query_devices(audio_index)
samplerate = int(device_info['default_samplerate'])
channels = 1 # TODO: query the device for the number of channels
# Set up serial port for AIOC adapter
# One can find the AIOC adapter by running "ls /dev/ttyACM*" for the com port, 
# and "aplay -l" for the audio device index.
serial_port = "/dev/ttyACM1" # TODO: query the operating system for the serial port
try:
    ser = serial.Serial(serial_port, timeout=1)  # Open serial port (adjust baud rate if needed)
    print("Serial port opened successfully.")
    ser.setRTS(False)
    ser.setDTR(False)
except Exception as e:
    print(f"[ERROR] Failed to open serial port: {e}")
    exit()

# Initialize Whisper voice recognition
model = whisper.load_model("small")

# Initialize CoquiTTS. TODO: find a better model or TTS engine that identifies and spells out acronyms.
coqui_tts_engine = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True, gpu=True)

### MAIN LOOP ###

# Initialize wake word detector (choose method)
# Option 1: Use "seven" with AST classification model (most efficient)
# Option 2: Use "Overlord" with custom detector (more flexible)
wake_detector = wake_word_detector.create_wake_word_detector("ast", device_sample_rate=samplerate)  # Change to "custom" for Overlord

print("🚀 Ham Radio AI Assistant starting up...")
print(f"Wake word detector: Ready")
print(f"Speech recognition: Whisper {model}")
print(f"Text-to-speech: CoquiTTS")
print(f"AI model: {MODEL}")
print("=" * 50)

while True:
    try:
        # Step 1: Wait for wake word detection
        print("\n🎤 Listening for wake word...")
        
        if isinstance(wake_detector, wake_word_detector.CustomWakeWordDetector):
            # Custom detector returns (detected, transcribed_text)
            wake_detected, initial_text = wake_detector.listen_for_wake_word(
                audio_device_index=audio_index, 
                samplerate=samplerate,
                debug=False
            )
            
            if not wake_detected:
                print("❌ Wake word not detected, continuing to listen...")
                continue
                
            print(f"✅ Wake word detected! Initial transcription: '{initial_text}'")
            
            # Check if this was just the wake word or includes a command
            if len(initial_text.split()) > 1:
                # Likely includes command after wake word
                operator_text = initial_text
                print("📝 Using initial transcription as full command")
            else:
                # Just wake word, need to get the actual command
                print("🎤 Wake word detected, now listening for your command...")
                operator_text = get_full_command_after_wake_word(audio_index, samplerate, channels, model)
                
        elif isinstance(wake_detector, wake_word_detector.ASTWakeWordDetector):
            # AST detector just returns boolean
            wake_detected = wake_detector.listen_for_wake_word(
                audio_device_index=audio_index,
                debug=False
            )
            
            if not wake_detected:
                print("❌ Wake word not detected, continuing to listen...")
                continue
                
            print("✅ Wake word 'seven' detected! Now listening for your command...")
            operator_text = get_full_command_after_wake_word(audio_index, samplerate, channels, model)
        
        else:
            # Fallback for other detector types
            wake_detected = wake_detector.listen_for_wake_word(
                audio_device_index=audio_index,
                debug=False
            )
            
            if not wake_detected:
                print("❌ Wake word not detected, continuing to listen...")
                continue
                
            print("✅ Wake word detected! Now listening for your command...")
            operator_text = get_full_command_after_wake_word(audio_index, samplerate, channels, model)
        
        # Step 2: Process the command

        # RICHC: This is a hack to get the wake word detector to pass the name of the bot.
        operator_text = f"{BOT_NAME}, {operator_text}"
        print(f"🗣️  Processing command: '{operator_text}'")
        
        # Check for termination command
        if re.search(rf"{re.escape(BOT_NAME)}.*?\b(break|brake|exit|quit|shutdown)\b", operator_text, re.IGNORECASE):
            print("🛑 Termination command detected. Shutting down...")
            play_tts_audio(f"Have a nice day! This is {BOT_PHONETIC_CALLSIGN} signing off. " +
                "I am clear and shutting down my processes.", coqui_tts_engine, ser, audio_index)
            break
        
        print(f"💬 Conversation request detected")
            
        # Ask AI: Send transcribed test from the operator to the AI
        current_prompt = prompts.add_operator_request_to_context(operator_text)
        print("🤖 Sending to Ollama...")
        print(f"Current prompt: {current_prompt}")
        ollama_response = ask_ollama(current_prompt)
        print(f"🤖 Ollama replied: {ollama_response}")
        prompts.add_ai_response_to_context(ollama_response)
            
        # Speak response
        print("🔊 Speaking response...")
        play_tts_audio(ollama_response, coqui_tts_engine, ser, audio_index)

            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Shutting down...")
        break
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
        continue

print("🏁 Ham Radio AI Assistant shutdown complete.")
ser.close()