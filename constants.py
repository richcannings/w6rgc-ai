# W6RGC-AI: Off Grid Ham Radio AI Voice Assistant
#
# Author: Rich Cannings <rcannings@gmail.com>
# License: Apache License, Version 2.0
# https://www.apache.org/licenses/LICENSE-2.0
#
# Description:
# This file centralizes all configuration values, thresholds, and constants
# used throughout the W6RGC-AI voice assistant application. It helps in
# managing settings for audio processing, wake word detection, hardware,
# AI/LLM integration, TTS, and other modules from a single location.
#
"""
Constants for the W6RGC AI Ham Radio Assistant

This file centralizes all configuration values, thresholds, and constants
used throughout the application for easier maintenance and configuration.
"""

# ============================================================================
# RADIO/HAM RADIO CONFIGURATION
# ============================================================================

# Bot Identity
OPERATOR_NAME = "Operator"
BOT_NAME = "7"
BOT_CALLSIGN = "W6RGC/AI"
BOT_SPOKEN_CALLSIGN = "W 6 R G C stroke I A"
BOT_PHONETIC_CALLSIGN = "Whiskey 6 Radio Golf Charlie Stroke Artificial Intelligence"

# ============================================================================
# AUDIO PROCESSING CONSTANTS
# ============================================================================

# Audio Detection Thresholds
AUDIO_THRESHOLD = 0.02  # Adjust as needed for your mic/environment
SILENCE_DURATION = 2.0  # seconds of silence to consider end of speech
FRAME_DURATION = 0.1   # seconds per audio frame

# Carrier Sense Configuration
CARRIER_SENSE_DURATION = 0.5  # seconds to monitor for carrier before PTT
CARRIER_SENSE_MAX_RETRIES = 3  # maximum attempts to find clear frequency
CARRIER_SENSE_RETRY_DELAY = 3.0  # seconds to wait between carrier sense attempts

# Periodic Identification Configuration
PERIODIC_ID_INTERVAL_MINUTES = 10  # minutes between automatic identification announcements

# Command Detection Configuration
MAX_COMMAND_WORDS = 10  # maximum words to check for commands (prevents accidental triggers)

# Audio Sample Rates
WHISPER_TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
DEFAULT_DEVICE_SAMPLE_RATE = 44100  # Standard audio device sample rate

# Audio Channels
DEFAULT_AUDIO_CHANNELS = 1  # Mono audio

# ============================================================================
# WAKE WORD DETECTION
# ============================================================================

# AST Wake Word Detection
DEFAULT_WAKE_WORD = "seven"
AST_CONFIDENCE_THRESHOLD = 0.7
AST_CHUNK_LENGTH_S = 1.0
AST_MODEL_NAME = "MIT/ast-finetuned-speech-commands-v2"

# Wake Word Detection Method
WAKE_WORD_METHOD_AST = "ast"
DEFAULT_WAKE_WORD_METHOD = WAKE_WORD_METHOD_AST

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

# Radio Interface Layer (RIL) Configuration
RIL_TYPE_AIOC = "aioc"
RIL_TYPE_DIGIRIG = "digirig"
DEFAULT_RIL_TYPE = "digirig" # RIL_TYPE_AIOC  # Change this to switch between AIOC and Digirig

# Serial Port Configuration
DEFAULT_AIOC_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_DIGIRIG_SERIAL_PORT = "/dev/ttyUSB2"
SERIAL_TIMEOUT = 1  # seconds

# ============================================================================
# AI/LLM CONFIGURATION
# ============================================================================

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:12b"  # Alternative: "gemma3:4b"

# ============================================================================
# TTS (Text-to-Speech) CONFIGURATION
# ============================================================================

# TTS Model Options
TTS_MODEL_FAST_PITCH = "tts_models/en/ljspeech/fast_pitch"
TTS_MODEL_SPEEDY_SPEECH = "tts_models/en/ljspeech/speedy_speech"
TTS_MODEL_TACOTRON2 = "tts_models/en/ljspeech/tacotron2-DDC"

# TTS Audio Settings
TTS_INFERENCE_NOISE_SCALE = 0.667
TTS_INFERENCE_NOISE_SCALE_DP = 1.0
TTS_INFERENCE_SIGMA = 1.0

# TTS File Configuration
TTS_OUTPUT_FILE = 'ollama_tts.wav'

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Prompt Types
PROMPT_TYPE_ORIGINAL = "original"
PROMPT_TYPE_RADIO_SCRIPT = "radio_script"
DEFAULT_PROMPT_TYPE = PROMPT_TYPE_ORIGINAL

# ============================================================================
# TESTING/DEBUG CONSTANTS
# ============================================================================

# Test Audio Generation
TEST_FREQUENCY = 440  # Hz (A4 note)
TEST_DURATION = 1.0   # seconds
TEST_AMPLITUDE = 0.5  # Audio amplitude for test signals

# Test TTS
TEST_TTS_TEXT = "This is a test of the text to speech system. How does it sound?"

# ============================================================================
# DEVICE DETECTION
# ============================================================================

# CUDA/GPU Configuration
CUDA_DEVICE = "cuda:0"
CPU_DEVICE = "cpu"

# ============================================================================
# FILE PATHS
# ============================================================================

# Temporary Files
TEMP_AUDIO_DIR = "/tmp"  # Could be made configurable

# ============================================================================
# PERFORMANCE TUNING
# ============================================================================

# Audio Processing
AUDIO_FRAME_MS = 100  # milliseconds per audio frame for processing 