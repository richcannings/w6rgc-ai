#!/usr/bin/env python3
# constants.py - Configuration Constants
#
# This file centralizes all configuration values, thresholds, and constants
# used throughout the W6RGC-AI voice assistant application. It helps in
# managing settings for audio processing, wake word detection, hardware,
# AI/LLM integration, TTS, and other modules from a single location.
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

from datetime import datetime

# ============================================================================
# Prompt / Chatbot Constants
# ============================================================================

# Bot Identity
BOT_NAME = "seven" # Here are the following wake word / bot name options: seven (7), marvin , shiela, zero (0), happy, forward.
BOT_CALLSIGN = "W6RGC/AI"
BOT_SPOKEN_CALLSIGN = "W 6 R G C stroke I A"
BOT_PHONETIC_CALLSIGN = "Whiskey 6 Romeo Golf Charlie Stroke Alpha India"

# User Identity
OPERATOR_NAME = "Operator"

# ============================================================================
# Hardware Configuration
# ============================================================================

# Radio Interface Layer (RIL) Configuration
RIL_TYPE_AIOC = "aioc"
RIL_TYPE_DIGIRIG = "digirig"
DEFAULT_RIL_TYPE = RIL_TYPE_DIGIRIG # Change this to switch between AIOC and Digirig

# Serial Port Configuration
DEFAULT_AIOC_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_DIGIRIG_SERIAL_PORT = "/dev/ttyUSB1"
SERIAL_TIMEOUT = 1  # seconds

# ============================================================================
# Speech Recognition / Audio Processing Constants
# ============================================================================

# Speech Recognition Model
WHISPER_MODEL = "medium.en"  # Whisper model size to use. Default is "small" or "small.en"

# Audio Sample Rates
WHISPER_TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
DEFAULT_DEVICE_SAMPLE_RATE = 44100  # Standard audio device sample rate

# Audio Detection Thresholds
AUDIO_THRESHOLD = 0.1  # Adjust as needed for your mic/environment (was 0.02)
SILENCE_DURATION = 1.0  # seconds of silence to consider end of speech (was 2.0)
FRAME_DURATION = 0.1   # seconds per audio frame

# Carrier Sense Configuration
CARRIER_SENSE_DURATION = 0.5  # seconds to monitor for carrier before PTT
CARRIER_SENSE_MAX_RETRIES = 3  # maximum attempts to find clear frequency
CARRIER_SENSE_RETRY_DELAY = 3.0  # seconds to wait between carrier sense attempts

# Periodic Identification Configuration
PERIODIC_ID_INTERVAL_MINUTES = 10  # minutes between automatic identification announcements

# Command Detection Configuration
MAX_COMMAND_WORDS = 10  # maximum words to check for commands (prevents accidental triggers)

# Audio Channels
DEFAULT_AUDIO_CHANNELS = 1  # Mono audio

# ============================================================================
# Wake Word Detection
# ============================================================================

# True means that the app will first listen for a transmission before listening for the wake word.
# False means that the app will listen for the wake word immediately, even when no one is transmitting.
DETECT_TRANSMISSION_BEFORE_WAKE_WORD = True

# AST Wake Word Detection
DEFAULT_WAKE_WORD = BOT_NAME # syncing bot name with wake word
AST_MODEL_NAME = "MIT/ast-finetuned-speech-commands-v2"
AST_CONFIDENCE_THRESHOLD = 0.7
AST_CHUNK_LENGTH_S = 1.0

# Wake Word Detection Method
WAKE_WORD_METHOD_AST = "ast"
DEFAULT_WAKE_WORD_METHOD = WAKE_WORD_METHOD_AST

# ============================================================================
# Intelligence Engine (LLM) Configuration
# ============================================================================

# Internet Connectivity and LLM Selection
HAS_INTERNET = True  # Set to True for Gemini (online), False for Ollama (offline)

# Ollama Configuration (offline)
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OFFLINE_MODEL = "gemma3:12b"  # Alternative: "gemma3:4b"

# Gemini Configuration (online)
GEMINI_API_KEY_FILE = "gemini_api_key.txt"
DEFAULT_ONLINE_MODEL = "models/gemini-2.5-flash-preview-05-20" # "gemini-1.5-flash" 
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
REQUEST_TIMEOUT = 30  # seconds


# ============================================================================
# TTS (Text-to-Speech) Configuration
# ============================================================================

# TTS Engine Selection
TTS_ENGINE_COQUI = "coqui"
TTS_ENGINE_PIPER = "piper"
DEFAULT_TTS_ENGINE = TTS_ENGINE_PIPER  # Options: "coqui" or "piper"

# Script File Path
WRITE_SCRIPT_TO_FILE = False

# Generate timestamped script file path
_now = datetime.now()
_time_str = _now.strftime("%H-%M-%S")
_date_str = _now.strftime("%Y-%m-%d")
SCRIPT_FILE_PATH = f"chatbot-script-{_date_str}-{_time_str}.log"

# Coqui TTS Model Options
TTS_MODEL_FAST_PITCH = "tts_models/en/ljspeech/fast_pitch"
TTS_MODEL_SPEEDY_SPEECH = "tts_models/en/ljspeech/speedy_speech"
TTS_MODEL_TACOTRON2 = "tts_models/en/ljspeech/tacotron2-DDC"

# Piper TTS Model Options
# You need to download the piper model files (.onnx and .onnx.json) manually
# from https://huggingface.co/rhasspy/piper-voices/tree/main
# and provide the full path to the .onnx file below.
TTS_PIPER_MODEL_PATH = "piper-tts-models/en_US-lessac-medium.onnx" # Example: "/path/to/your/piper/voices/en_US-lessac-medium.onnx"

# TTS Audio Settings
TTS_INFERENCE_NOISE_SCALE = 0.667
TTS_INFERENCE_NOISE_SCALE_DP = 1.0
TTS_INFERENCE_SIGMA = 1.0

# TTS File Configuration
TTS_OUTPUT_FILE = 'ollama_tts.wav'

# ============================================================================
# Testing / Debug Constants
# ============================================================================

# Test Audio Generation
TEST_FREQUENCY = 440  # Hz (A4 note)
TEST_DURATION = 1.0   # seconds
TEST_AMPLITUDE = 0.5  # Audio amplitude for test signals

# Test TTS
TEST_TTS_TEXT = "This is a test of the text to speech system. How does it sound?"

# ============================================================================
# GPU/CPU Device Detection
# ============================================================================

# CUDA/GPU Configuration
CUDA_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"

# ============================================================================
# File Paths
# ============================================================================

# Temporary Files
TEMP_AUDIO_DIR = "/tmp"  # Could be made configurable

# ============================================================================
# Performance Tuning
# ============================================================================

# Audio Processing
AUDIO_FRAME_MS = 100  # milliseconds per audio frame for processing 

# Carrier sense before wake word detection
DETECT_TRANSMISSION_BEFORE_WAKE_WORD = True
