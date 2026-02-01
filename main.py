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
#  - Radio interface adapter support for PTT control
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

import numpy as np
import time
import gc
import threading
import sounddevice as sd
import librosa
from scipy.signal import resample
from TTS.api import TTS as CoquiTTS

# modules part of the w6rgc/ai project
import wake_word_detector
import regex_command_tooling
from speech_recognition import SpeechRecognitionEngine
from context_manager import ContextManager
from ril_aioc import RadioInterfaceLayerAIOC
from ril_digirig import RadioInterfaceLayerDigiRig
from periodically_identify import PeriodicIdentifier
from llm_gemini_online import ask_gemini
from llm_ollama_offline import ask_ollama
from llm_openclaw_local import ask_openclaw
from piper_tts_wrapper import PiperTTSWrapper
from audio_level_monitor import AudioLevelMonitor

### CONSTANTS ###

from constants import (
    # Bot name and phonetic call sign
    BOT_NAME,
    BOT_PHONETIC_CALLSIGN,
    
    # Speech recognition configuration
    WHISPER_MODEL,
    AUDIO_THRESHOLD,
    FRAME_DURATION,
    WHISPER_TARGET_SAMPLE_RATE,
    
    # AI/LLM configuration
    HAS_INTERNET,
    LLM_ENGINE,
    LLM_ENGINE_AUTO,
    LLM_ENGINE_GEMINI,
    LLM_ENGINE_OLLAMA,
    LLM_ENGINE_OPENCLAW,
    DEFAULT_OFFLINE_MODEL,
    DEFAULT_ONLINE_MODEL,
    OPENCLAW_AGENT_ID,
    
    # TTS configuration
    DEFAULT_TTS_ENGINE,
    TTS_ENGINE_COQUI,
    TTS_ENGINE_PIPER,
    TTS_PIPER_MODEL_PATH,
    TTS_MODEL_TACOTRON2,
    TTS_INFERENCE_NOISE_SCALE,
    TTS_INFERENCE_NOISE_SCALE_DP,
    TTS_INFERENCE_SIGMA,
    
    # Hardware configuration
    DEFAULT_AIOC_SERIAL_PORT,
    DEFAULT_DIGIRIG_SERIAL_PORT,
    RIL_TYPE_AIOC,
    RIL_TYPE_DIGIRIG,
    DEFAULT_RIL_TYPE,
    
    # Wake word detection
    WAKE_WORD_METHOD_AST,
    AST_MODEL_NAME,
    DEFAULT_WAKE_WORD,
    DETECT_TRANSMISSION_BEFORE_WAKE_WORD,
    AST_CONFIDENCE_THRESHOLD,
    AST_CHUNK_LENGTH_S,
    
    # Audio level monitoring
    ENABLE_AUDIO_LEVEL_MONITORING
)

### HELPER CLASSES AND FUNCTIONS ###

def parallel_wake_word_and_speech_recognition(wake_detector, speech_recognition_engine, audio_index, wake_word_timeout=5, max_recording_duration=60):
    """
    Parallel wake word detection and speech recognition with optional audio level monitoring.
    
    Both processes start simultaneously and share the same audio stream.
    If wake word is detected, speech recognition result is returned.
    If wake word is not detected, returns None for both.
    
    Audio level monitoring can be enabled/disabled via ENABLE_AUDIO_LEVEL_MONITORING constant.
    
    Args:
        wake_detector: Wake word detector instance
        speech_recognition_engine: Speech recognition engine instance
        audio_index: Audio device index
        wake_word_timeout: Maximum duration to listen for wake word (seconds)
        max_recording_duration: Maximum total recording duration (seconds)
        
    Returns:
        tuple: (wake_detected: bool, transcribed_text: str or None)
    """
    print(f"🎤 Starting parallel wake word detection and speech recognition...")
    
    # Initialize audio level monitor (if enabled)
    audio_monitor = AudioLevelMonitor() if ENABLE_AUDIO_LEVEL_MONITORING else None
    
    # Shared state between threads
    wake_detected = threading.Event()
    speech_complete = threading.Event()
    stop_recording = threading.Event()
    
    # Results
    transcribed_text = None
    wake_word_found = False
    recording_start_time = None  # Track when recording actually starts
    
    # Get stream parameters from RIL
    stream_params = speech_recognition_engine.ril_interface.get_input_stream_params()
    samplerate = stream_params['samplerate']
    
    # Audio buffers for both processes
    wake_word_buffer = []
    speech_buffer = []
    full_speech_recording = []  # Preserve all speech audio for final transcription
    speech_started = False
    silence_counter = 0
    
    def wake_word_thread():
        """Thread for wake word detection"""
        nonlocal wake_word_found
        
        # Calculate chunk size for wake word detection
        chunk_samples = int(AST_CHUNK_LENGTH_S * samplerate)
        wake_word_model_sample_rate = wake_detector.model_sample_rate
        
        print(f"🎯 Wake word thread started (listening for '{DEFAULT_WAKE_WORD}')")
        
        while not stop_recording.is_set() and not wake_detected.is_set():
            if len(wake_word_buffer) >= chunk_samples:
                # Extract chunk for analysis
                audio_chunk = np.array(wake_word_buffer[:chunk_samples])
                wake_word_buffer[:] = wake_word_buffer[chunk_samples:]  # Remove processed samples
                
                # Resample if needed
                if samplerate != wake_word_model_sample_rate:
                    audio_chunk = librosa.resample(audio_chunk, 
                                                 orig_sr=samplerate, 
                                                 target_sr=wake_word_model_sample_rate)
                
                # Run classification
                try:
                    prediction = wake_detector.classifier(audio_chunk, sampling_rate=wake_word_model_sample_rate)
                    if isinstance(prediction, list) and len(prediction) > 0:
                        prediction = prediction[0]  # Get top prediction
                    
                    # Check if wake word detected with sufficient confidence
                    if (prediction["label"].lower() == wake_detector.wake_word and 
                        prediction["score"] > AST_CONFIDENCE_THRESHOLD):
                        
                        print(f"🎯 Wake word '{DEFAULT_WAKE_WORD}' detected! (confidence: {prediction['score']:.3f})")
                        wake_word_found = True
                        wake_detected.set()
                        return
                        
                except Exception as e:
                    print(f"❌ Wake word detection error: {e}")
            else:
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        print("🎯 Wake word thread finished")
    
    def speech_recognition_thread():
        """Thread for speech recognition"""
        nonlocal transcribed_text, speech_started, silence_counter
        
        print("📝 Speech recognition thread started")
        
        # Use longer silence duration for extended conversations
        extended_silence_duration = 3.0  # 3 seconds of silence for longer conversations
        
        while not stop_recording.is_set():
            if len(speech_buffer) >= int(FRAME_DURATION * samplerate):
                # Extract frame for speech detection
                frame_samples = int(FRAME_DURATION * samplerate)
                frame = np.array(speech_buffer[:frame_samples])
                
                # Always preserve speech audio in full_speech_recording when we have speech
                if speech_started or np.max(np.abs(frame)) > AUDIO_THRESHOLD:
                    full_speech_recording.extend(frame)
                
                speech_buffer[:] = speech_buffer[frame_samples:]  # Remove processed samples
                
                amplitude = np.max(np.abs(frame))
                
                if amplitude > AUDIO_THRESHOLD:
                    if not speech_started:
                        print("🗣️  Speech detected, recording...")
                        speech_started = True
                    silence_counter = 0
                elif speech_started:
                    silence_counter += FRAME_DURATION
                    # Use longer silence duration for extended conversations
                    if silence_counter >= extended_silence_duration:
                        print(f"🔇 Extended silence detected ({extended_silence_duration}s), processing...")
                        speech_complete.set()
                        break
            else:
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        print("📝 Speech recognition thread finished")
    
    # Reset audio device
    speech_recognition_engine.ril_interface.reset_audio_device()
    
    # Start both threads
    wake_thread = threading.Thread(target=wake_word_thread, daemon=True)
    speech_thread = threading.Thread(target=speech_recognition_thread, daemon=True)
    
    start_time = time.time()
    
    try:
        wake_thread.start()
        speech_thread.start()
        
        # Main audio capture loop with level monitoring
        with sd.InputStream(**stream_params) as stream:
            frame_counter = 0
            while True:
                elapsed_time = time.time() - start_time
                
                # Check wake word timeout (only before wake word is detected)
                if not wake_detected.is_set() and elapsed_time > wake_word_timeout:
                    print(f"⌛ Timeout: Did not detect '{DEFAULT_WAKE_WORD}' within {wake_word_timeout} seconds.")
                    stop_recording.set()
                    break
                
                # Check if wake word was detected
                if wake_detected.is_set():
                    if recording_start_time is None:
                        recording_start_time = time.time()
                        print("✅ Wake word detected, continuing speech recognition...")
                    
                    # Check maximum recording duration
                    if recording_start_time and (time.time() - recording_start_time) > max_recording_duration:
                        print(f"⌛ Maximum recording duration ({max_recording_duration}s) reached. Processing...")
                        stop_recording.set()
                        speech_complete.set()
                        break
                    
                    # Continue recording until speech is complete
                    while not speech_complete.is_set() and not stop_recording.is_set():
                        frame, _ = stream.read(int(FRAME_DURATION * samplerate))
                        frame = np.squeeze(frame)
                        speech_buffer.extend(frame)
                        # Also preserve for final transcription
                        if speech_started:
                            full_speech_recording.extend(frame)
                        
                        # Monitor audio levels during continued recording (if enabled)
                        if audio_monitor:
                            audio_monitor.add_frame(frame)
                        
                        # Check max recording duration again during continued recording
                        if recording_start_time and (time.time() - recording_start_time) > max_recording_duration:
                            print(f"⌛ Maximum recording duration ({max_recording_duration}s) reached during continued recording.")
                            stop_recording.set()
                            speech_complete.set()
                            break
                        
                        time.sleep(0.01)
                    break
                
                # Read audio and feed to both buffers
                frame, _ = stream.read(int(FRAME_DURATION * samplerate))
                frame = np.squeeze(frame)
                
                # Monitor audio levels (if enabled)
                if audio_monitor:
                    audio_monitor.add_frame(frame)
                frame_counter += 1
                
                # Show instantaneous levels periodically (if enabled)
                if ENABLE_AUDIO_LEVEL_MONITORING and frame_counter % 20 == 0:  # Every 2 seconds (20 frames * 0.1s)
                    peak = np.max(np.abs(frame))
                    rms = np.sqrt(np.mean(frame**2))
                    if wake_detected.is_set() and recording_start_time:
                        recording_duration = time.time() - recording_start_time
                        print(f"🔊 Recording {recording_duration:.1f}s | levels: peak={peak:.3f}, rms={rms:.3f}, threshold={AUDIO_THRESHOLD:.3f}")
                    else:
                        print(f"🔊 Instant levels: peak={peak:.3f}, rms={rms:.3f}, threshold={AUDIO_THRESHOLD:.3f}")
                
                # Feed to both buffers
                wake_word_buffer.extend(frame)
                speech_buffer.extend(frame)
                
                # Also preserve speech audio if we detect speech activity
                amplitude = np.max(np.abs(frame))
                if speech_started or amplitude > AUDIO_THRESHOLD:
                    full_speech_recording.extend(frame)
                    if not speech_started and amplitude > AUDIO_THRESHOLD:
                        speech_started = True
                        print(f"🗣️  Main loop: Speech detected (amplitude: {amplitude:.3f})")
    
    except Exception as e:
        print(f"❌ Audio capture error: {e}")
        stop_recording.set()
    
    finally:
        # Ensure threads complete
        stop_recording.set()
        wake_thread.join(timeout=1.0)
        speech_thread.join(timeout=1.0)
        
        # Final audio level report (if enabled)
        if audio_monitor:
            stats = audio_monitor.get_current_stats()
            if stats:
                print(f"📊 Final Audio Statistics:")
                print(f"   Processed {stats['frame_count']} frames")
                print(f"   Peak: avg={stats['avg_peak']:.3f}, max={stats['max_peak']:.3f}")
                print(f"   RMS:  avg={stats['avg_rms']:.3f}, max={stats['max_rms']:.3f}")
    
    # Process speech recognition if wake word was detected
    if wake_word_found:
        total_recording_time = (time.time() - recording_start_time) if recording_start_time else 0
        print(f"📝 Processing recorded speech... (speech_started: {speech_started}, buffer size: {len(full_speech_recording)} samples)")
        print(f"📝 Total recording time: {total_recording_time:.1f} seconds")
        
        # Reconstruct the full audio from the complete recording
        if full_speech_recording:
            audio = np.array(full_speech_recording)
            if audio.ndim > 1:
                audio = audio[:, 0]
            
            print(f"📝 Audio data: {len(audio)} samples, duration: {len(audio)/samplerate:.2f}s")
            
            # Final audio level analysis for the speech segment (if enabled)
            if ENABLE_AUDIO_LEVEL_MONITORING and len(audio) > 0:
                speech_peak = np.max(np.abs(audio))
                speech_rms = np.sqrt(np.mean(audio**2))
                print(f"📝 Speech segment levels: peak={speech_peak:.3f}, rms={speech_rms:.3f}")
                
                if speech_peak < 0.05:
                    print("⚠️  Speech segment very quiet - consider increasing input gain")
                elif speech_peak > 0.95:
                    print("⚠️  Speech segment clipping detected - consider decreasing input gain")
            
            # Resample if needed for Whisper
            if samplerate != WHISPER_TARGET_SAMPLE_RATE:
                num_samples = int(len(audio) * WHISPER_TARGET_SAMPLE_RATE / samplerate)
                audio = resample(audio, num_samples)
                print(f"📝 Resampled to: {len(audio)} samples for Whisper")
            
            # Transcribe with Whisper
            try:
                print("📝 Transcribing with Whisper...")
                result = speech_recognition_engine.model.transcribe(audio, fp16=True, language='en')
                transcribed_text = result['text'].strip()
                print(f"📝 Transcribed: '{transcribed_text}'")
            except Exception as e:
                print(f"❌ Transcription error: {e}")
                transcribed_text = ""
        else:
            print(f"❌ No speech audio to process (speech_started: {speech_started}, buffer_size: {len(full_speech_recording)})")
            transcribed_text = ""
    else:
        # Wake word not found, no transcription
        transcribed_text = None
    
    return wake_word_found, transcribed_text

def create_tts_engine():
    """
    Factory function to create the appropriate TTS engine instance.
    """
    if DEFAULT_TTS_ENGINE == TTS_ENGINE_COQUI:
        print("🔧 Initializing Coqui TTS engine...")
        try:
            engine = CoquiTTS(model_name=TTS_MODEL_TACOTRON2, progress_bar=True, gpu=True)
            print(f"✅ Using Coqui TTS model: {TTS_MODEL_TACOTRON2}")
            # Configure CoquiTTS for speed over quality
            if hasattr(engine, 'synthesizer') and hasattr(engine.synthesizer, 'tts_config'):
                try:
                    engine.synthesizer.tts_config.inference_noise_scale = TTS_INFERENCE_NOISE_SCALE
                    engine.synthesizer.tts_config.inference_noise_scale_dp = TTS_INFERENCE_NOISE_SCALE_DP
                    engine.synthesizer.tts_config.inference_sigma = TTS_INFERENCE_SIGMA
                    print("✅ Coqui TTS speed optimizations applied")
                except:
                    print("⚠️  Could not apply Coqui TTS speed optimizations")
            return engine
        except Exception as e:
            print(f"❌ CRITICAL: Failed to initialize Coqui TTS model ({TTS_MODEL_TACOTRON2}): {e}")
            return None
            
    elif DEFAULT_TTS_ENGINE == TTS_ENGINE_PIPER:
        print("🔧 Initializing Piper TTS engine...")
        try:
            engine = PiperTTSWrapper(model_path=TTS_PIPER_MODEL_PATH)
            print(f"✅ Using Piper TTS model: {TTS_PIPER_MODEL_PATH}")
            return engine
        except Exception as e:
            print(f"❌ CRITICAL: Failed to initialize Piper TTS model ({TTS_PIPER_MODEL_PATH}): {e}")
            return None
    else:
        raise ValueError(f"Unsupported TTS engine type: {DEFAULT_TTS_ENGINE}")

def create_radio_interface_layer(ril_type=DEFAULT_RIL_TYPE, serial_port_name=None):
    """
    Factory function to create the appropriate Radio Interface Layer instance.
    
    Args:
        ril_type (str): Type of RIL to create ("aioc" or "digirig")
        serial_port_name (str): Serial port for PTT control (auto-selected if None)
        
    Returns:
        RadioInterfaceLayer instance (AIOC or Diggirig)
        
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

def play_tts_audio(text_to_speak, tts_engine, radio_interface):
    """
    In memory TTS audio playback
    Handles PTT, audio preparation, and playback through the radio interface.
    
    Args:
        text_to_speak (str): Text to convert to speech
        tts_engine: CoquiTTS engine instance
        radio_interface: Radio interface layer instance
    """
    ptt_activated = False
    try:
        # Reset audio device and generate TTS
        radio_interface.reset_audio_device()
        tts_audio_data = tts_engine.tts(text=text_to_speak)
        tts_sample_rate = tts_engine.synthesizer.output_sample_rate
        
        # Convert and normalize audio data
        if not isinstance(tts_audio_data, np.ndarray):
            tts_audio_data = np.array(tts_audio_data, dtype=np.float32)
        
        # Convert to mono if needed
        if tts_audio_data.ndim > 1 and tts_audio_data.shape[1] > 1:
            tts_audio_data = np.mean(tts_audio_data, axis=1)
        
        # Normalize audio
        max_val = np.max(np.abs(tts_audio_data))
        if max_val > 0:
            tts_audio_data = tts_audio_data * 0.95 / max_val
        
        # Transmit audio - CRITICAL: Track PTT state
        radio_interface.ptt_on()
        ptt_activated = True
        
        # Play audio with additional error handling
        try:
            radio_interface.play_audio(tts_audio_data, tts_sample_rate)
        except Exception as play_error:
            print(f"❌ Audio playback error: {play_error}")
            # Don't re-raise, we still need to turn PTT OFF
        
    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
    finally:
        # CRITICAL: Always turn PTT OFF, even if there are exceptions
        if ptt_activated:
            try:
                radio_interface.ptt_off()
            except Exception as ptt_error:
                print(f"❌ CRITICAL: PTT OFF failed: {ptt_error}")
                # Try emergency PTT OFF
                try:
                    if hasattr(radio_interface, 'serial_conn') and radio_interface.serial_conn and radio_interface.serial_conn.is_open:
                        radio_interface.serial_conn.setRTS(False)
                        print("🚨 Emergency PTT OFF executed")
                except Exception as emergency_error:
                    print(f"❌ EMERGENCY PTT OFF FAILED: {emergency_error}")
        
        # Cleanup
        if 'tts_audio_data' in locals():
            del tts_audio_data
        gc.collect()

### INITIALIZATION ###

# Initialize the Radio Interface Layer
try:
    print(f"🔧 Selected RIL type: {DEFAULT_RIL_TYPE}")
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
speech_recognition_engine = SpeechRecognitionEngine(ril)

# Initialize TTS Engine
tts_engine = create_tts_engine()
if tts_engine is None:
    print("TTS will not be available. Exiting.")
    exit()

# Initialize wake word detector (using AST method)
wake_detector = wake_word_detector.create_wake_word_detector(
    method=WAKE_WORD_METHOD_AST, 
    device_sample_rate=samplerate,
    wake_word=DEFAULT_WAKE_WORD
)

# Initialize periodic identifier
#periodic_identifier = PeriodicIdentifier(
#    tts_engine=tts_engine,
#    radio_interface=ril,
#    play_tts_function=play_tts_audio  # Use the fast method
#)
#periodic_identifier.start()

# Print startup information
print("🚀 Ham radio AI voice assistant starting up...")
print(f"Wake word detector: Ready (AST method, wake word: '{DEFAULT_WAKE_WORD}')")
print(f"Speech recognition: Whisper model: {WHISPER_MODEL}")
startup_engine = LLM_ENGINE
if startup_engine == LLM_ENGINE_AUTO:
    startup_engine = LLM_ENGINE_GEMINI if HAS_INTERNET else LLM_ENGINE_OLLAMA

if startup_engine == LLM_ENGINE_GEMINI:
    print(f"AI model: {DEFAULT_ONLINE_MODEL} (online)")
elif startup_engine == LLM_ENGINE_OPENCLAW:
    print(f"AI model: OpenClaw agent {OPENCLAW_AGENT_ID} (local)")
else:
    print(f"AI model: {DEFAULT_OFFLINE_MODEL} (offline)")
print(f"Text-to-speech: {tts_engine.model_name}")
print(f"Audio level monitoring: {'Enabled' if ENABLE_AUDIO_LEVEL_MONITORING else 'Disabled'}")
print("=" * 50)

### MAIN LOOP ###

while True:
    try:
        # STEP #0: Listen for carrier signal. This needs to be a clear transtion from no transmission to a
        # transmission.
        if DETECT_TRANSMISSION_BEFORE_WAKE_WORD:
            # First, wait for the channel to be clear.
            while ril.check_carrier_sense(duration=0.1):
                print("🎤 Channel is busy. Waiting for it to be clear...")
                time.sleep(0.5) # Wait before re-checking

            # Now, wait for a new transmission to start.
            print("🎤 Channel is clear. Standing by for new transmission...")
            while not ril.check_carrier_sense(duration=0.1):
                time.sleep(0.1) # Poll frequently to catch the start
            print("🎤 New transmission detected.")
            # At this point, a transmission has started on a previously clear channel.
            
        # STEP 1: Parallel wake word detection and speech recognition
        print(f"🎤 Transmission detected. Starting parallel wake word detection and speech recognition...")
        wake_detected, operator_text = parallel_wake_word_and_speech_recognition(
            wake_detector=wake_detector,
            speech_recognition_engine=speech_recognition_engine,
            audio_index=audio_index,
            wake_word_timeout=5,  # Listen for 5 seconds for the wake word 
            max_recording_duration=60  # Allow up to 60 seconds of recording
        )
        
        if not wake_detected:
            print("❌ Wake word not detected. Restarting loop.")
            continue

        # Step 2a: Transmit a tone, notifying the operator that the chatbot copied their messsage.
        # TODO(richc): Kick off a thread to play notification audio clip to notify the user that 
        # the bot got the message. Tried. There is a delay... Just as much as it takes for the 
        # chatbot to response. I may dive int it later. I may need to implement a queue for the ril.

        print(f"✅ Wake word '{DEFAULT_WAKE_WORD}' detected! Speech recognition completed.")
        
        # Validate transcription result
        if operator_text is None:
            print("❌ Transcription failed. Restarting loop.")
            continue
        elif operator_text.strip() == "":
            print("❌ No speech detected after wake word. Restarting loop.")
            continue
        
        # STEP 2: Process the operator's input (Assumes the wake word is the same as the bots name.)

        # This is a hack to get the wake word detector to pass the name of the bot, so the bot
        # receives the entire transmission. This may or may not be valuable.
        # operator_text = f"{BOT_NAME}, {operator_text}"
        
        print(f"🗣️  Processing command: '{operator_text}'")
        
        # Optional step: Identify the command using the commands module
        # Rudimentary command handling. TODO(richc): Complete refactor based on AI tooling/function calling
        command_type = regex_command_tooling.handle_command(operator_text)

        if command_type == "terminate":
            print("🛑 Termination command identified by main.py. Shutting down...")
            play_tts_audio(f"Terminating. Have a nice day! This is {BOT_PHONETIC_CALLSIGN} shutting down my " +
                           "processes. I am clear. Seven three.", tts_engine, ril)
            break
        elif command_type == "status":
            print("⚙️ Status command identified by main.py.")
            if LLM_ENGINE == LLM_ENGINE_AUTO:
                if HAS_INTERNET:
                    llm_info = f"{DEFAULT_ONLINE_MODEL} online large language model"
                    internet_info = "I am connected to the internet."
                else:
                    llm_info = f"the {DEFAULT_OFFLINE_MODEL} offline large language model"
                    internet_info = "I am not connected to the internet."
            elif LLM_ENGINE == LLM_ENGINE_GEMINI:
                llm_info = f"{DEFAULT_ONLINE_MODEL} online large language model"
                internet_info = "I am connected to the internet."
            elif LLM_ENGINE == LLM_ENGINE_OLLAMA:
                llm_info = f"the {DEFAULT_OFFLINE_MODEL} offline large language model"
                internet_info = "I am not connected to the internet."
            else:
                llm_info = f"the OpenClaw agent {OPENCLAW_AGENT_ID}"
                internet_info = "I am connected to the local OpenClaw gateway."

            status_report = f"""I am {BOT_NAME}. All systems are go. {internet_info} I use:
                {llm_info} for intelligence, 
                the {AST_MODEL_NAME} for wake word detection, 
                the Whisper version {speech_recognition_engine.version} for speech recognition, and 
                the {tts_engine.model_name} for text-to-speech."""
            play_tts_audio(status_report, tts_engine, ril)
            continue
        elif command_type == "reset":
            print("🔄 Reset command identified by main.py. Resetting context.")
            context_mgr.reset_context()
            play_tts_audio("My context has been reset. I am ready for a new conversation.", 
                           tts_engine, ril)
            continue
        else:
            # STEP 3: If no command was handled, proceed with conversation between the operator and the bot
            print(f"💬 Conversation request detected")
            
            # Add the operator's request to the context
            current_prompt = context_mgr.add_operator_request_to_context(operator_text)
            
            # Choose LLM based on engine selection
            # TODO(richc): Decide which LLM to during initialization
            selected_engine = LLM_ENGINE
            if selected_engine == LLM_ENGINE_AUTO:
                selected_engine = LLM_ENGINE_GEMINI if HAS_INTERNET else LLM_ENGINE_OLLAMA

            if selected_engine == LLM_ENGINE_GEMINI:
                print("🤖 Sending to Gemini...")
                print(f"Current prompt: {current_prompt}")
                ai_response = ask_gemini(current_prompt)
                print(f"🤖 Gemini replied: {ai_response}")

                # Check for direct TTS command from Gemini function call
                if ai_response.startswith("TTS_DIRECT:"):
                    tts_message = ai_response.replace("TTS_DIRECT:", "", 1)
                    print(f"🔊 APRS - Speaking directly: {tts_message[:100]}...")
                    context_mgr.add_ai_response_to_context(tts_message) # Add tooling responses to script/context
                    play_tts_audio(tts_message, tts_engine, ril)
                    #periodic_identifier.restart_timer()
                    continue # Skip further processing of this response in the main loop
            elif selected_engine == LLM_ENGINE_OPENCLAW:
                print("🤖 Sending to OpenClaw...")
                print(f"Current prompt: {current_prompt}")
                ai_response = ask_openclaw(current_prompt)
                print(f"🤖 OpenClaw replied: {ai_response}")
            else:
                print("🤖 Sending to Ollama...")
                print(f"Current prompt: {current_prompt}")
                ai_response = ask_ollama(current_prompt)
                print(f"🤖 Ollama replied: {ai_response}")
                
            # add the AI's response to the context
            context_mgr.add_ai_response_to_context(ai_response)
            
            # STEP 4: Speak response
            print("🔊 Speaking response...")
            play_tts_audio(ai_response, tts_engine, ril)
            #periodic_identifier.restart_timer()

        # Reset audio device after TTS for next recording cycle
        ril.reset_audio_device()            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Shutting down...")
        break
    except Exception as e:
        print(f"❌ Error in main loop: {e}")
        continue

### SHUTDOWN ###

#periodic_identifier.stop() # Stop periodic identification
ril.close() # Close the RIL (which closes serial conn)
print("🏁 Ham Radio AI Assistant shutdown complete.")
