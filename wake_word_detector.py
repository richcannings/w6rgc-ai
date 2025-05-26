#!/usr/bin/env python3
# wake_word_detector.py - Advanced wake word detection for ham radio AI assistant
#
# This module provides dual wake word detection methods for the K6DIT-AI system:
#
# 1. AST Method (Recommended for efficiency):
#    - Uses MIT/ast-finetuned-speech-commands-v2 model
#    - 35+ pre-trained wake words available
#    - Very fast, low CPU usage, high accuracy
#    - Current default: "seven"
#    - Available words: backward, bed, bird, cat, dog, down, eight, five, follow, 
#      forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, 
#      one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero
#
# 2. Custom Method (Flexible):
#    - Uses energy detection + Whisper verification
#    - Can detect any custom phrase
#    - Higher CPU usage but more flexible
#    - Current default: "Overlord"
#
# Features:
#  - Automatic sample rate conversion (48kHz device → 16kHz model)
#  - CUDA acceleration when available
#  - Configurable confidence thresholds
#  - Debug mode for testing and tuning
#  - Graceful error handling
#
# Author: Rich Cannings, W6RGC
# Copyright 2025 Rich Cannings

import numpy as np
import sounddevice as sd
from transformers import pipeline
import torch
import time
import librosa
from typing import Optional, Tuple
from prompts import BOT_NAME

class ASTWakeWordDetector:
    """
    Efficient wake word detector using MIT/ast-finetuned-speech-commands-v2 model.
    This model is specifically trained on speech commands and is much more efficient
    than running Whisper continuously.
    """
    
    def __init__(self, 
                 wake_word: str = "seven",
                 confidence_threshold: float = 0.7,
                 chunk_length_s: float = 1.0,
                 device_sample_rate: int = 44100):
        """
        Initialize the AST wake word detector.
        
        Args:
            wake_word: The wake word to detect (must be in model's vocabulary)
            confidence_threshold: Minimum confidence to trigger wake word
            chunk_length_s: Length of audio chunks to analyze
            device_sample_rate: Sample rate of the audio device
        """
        self.wake_word = wake_word.lower()
        self.confidence_threshold = confidence_threshold
        self.chunk_length_s = chunk_length_s
        self.device_sample_rate = device_sample_rate
        
        # Initialize the audio classification pipeline
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading AST wake word detector on {device}...")
        
        self.classifier = pipeline(
            "audio-classification", 
            model="MIT/ast-finetuned-speech-commands-v2", 
            device=device
        )
        
        # Get model's expected sample rate
        self.model_sample_rate = self.classifier.feature_extractor.sampling_rate
        
        # Verify wake word is in model vocabulary
        self._verify_wake_word()
        
        print(f"AST wake word detector ready. Listening for: '{self.wake_word}'")
        print(f"Device sample rate: {self.device_sample_rate}Hz, Model sample rate: {self.model_sample_rate}Hz")
    
    def _verify_wake_word(self):
        """Verify that the wake word is in the model's vocabulary"""
        labels = list(self.classifier.model.config.id2label.values())
        labels_lower = [label.lower() for label in labels]
        
        if self.wake_word not in labels_lower:
            print(f"Available wake words: {labels}")
            raise ValueError(f"Wake word '{self.wake_word}' not found in model vocabulary: {labels}")
        
        print(f"✓ Wake word '{self.wake_word}' found in model vocabulary")
    
    def listen_for_wake_word(self, audio_device_index: int, debug: bool = False) -> bool:
        """
        Listen continuously for the wake word.
        Returns True when wake word is detected.
        
        Args:
            audio_device_index: Audio device to use for recording
            debug: Print classification results for debugging
        """
        print(f"🎤 Listening for wake word '{self.wake_word}'...")
        if debug:
            print("Debug mode: showing all predictions")
            print(f"Model expects {self.model_sample_rate}Hz, device uses {self.device_sample_rate}Hz")
        
        # Calculate chunk size for device sample rate
        chunk_samples = int(self.chunk_length_s * self.device_sample_rate)
        
        try:
            with sd.InputStream(samplerate=self.device_sample_rate, channels=1, 
                              device=audio_device_index, dtype='float32') as stream:
                
                while True:
                    # Read audio chunk
                    audio_chunk, _ = stream.read(chunk_samples)
                    audio_chunk = np.squeeze(audio_chunk)
                    
                    # Resample if needed
                    if self.device_sample_rate != self.model_sample_rate:
                        audio_chunk = librosa.resample(audio_chunk, 
                                                     orig_sr=self.device_sample_rate, 
                                                     target_sr=self.model_sample_rate)
                    
                    # Run classification
                    prediction = self.classifier(audio_chunk, sampling_rate=self.model_sample_rate)
                    if isinstance(prediction, list) and len(prediction) > 0:
                        prediction = prediction[0]  # Get top prediction
                    
                    if debug:
                        print(f"Prediction: {prediction['label']} ({prediction['score']:.3f})")
                    
                    # Check if wake word detected with sufficient confidence
                    if (prediction["label"].lower() == self.wake_word and 
                        prediction["score"] > self.confidence_threshold):
                        
                        print(f"🎯 Wake word '{self.wake_word}' detected! (confidence: {prediction['score']:.3f})")
                        return True
                        
        except KeyboardInterrupt:
            print("\n⏹️  Wake word detection stopped by user")
            return False
        except Exception as e:
            print(f"❌ Error in wake word detection: {e}")
            return False

class CustomWakeWordDetector:
    """
    Custom wake word detector using energy detection + Whisper verification.
    This is more flexible for custom wake words like "Overlord" that aren't
    in the AST model's vocabulary.
    """
    
    def __init__(self, 
                 wake_phrase: str = BOT_NAME,
                 energy_threshold: float = 0.02,
                 silence_duration: float = 1.5,
                 device_sample_rate: int = 44100):
        """
        Initialize custom wake word detector.
        
        Args:
            wake_phrase: The phrase to detect (will use Whisper for verification)
            energy_threshold: Voice activity detection threshold
            silence_duration: Seconds of silence to end recording
            device_sample_rate: Sample rate of the audio device
        """
        self.wake_phrase = wake_phrase.lower()
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.device_sample_rate = device_sample_rate
        
        # Load a small, fast Whisper model for verification
        import whisper
        print("Loading lightweight Whisper model for wake word verification...")
        self.whisper_model = whisper.load_model("tiny")
        print(f"Custom wake word detector ready for: '{wake_phrase}'")
    
    def listen_for_wake_word(self, 
                           audio_device_index: int, 
                           samplerate: int = None,  # For compatibility
                           debug: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Listen for wake word using energy detection + Whisper verification.
        
        Returns:
            (wake_word_detected, transcribed_text)
        """
        # Use device sample rate if not specified
        if samplerate is None:
            samplerate = self.device_sample_rate
            
        print(f"🎤 Listening for wake phrase '{self.wake_phrase}'...")
        
        recording = []
        speech_started = False
        silence_counter = 0
        frame_duration = 0.1  # 100ms frames
        
        with sd.InputStream(samplerate=samplerate, channels=1, 
                          device=audio_device_index, dtype='float32') as stream:
            
            while True:
                frame, _ = stream.read(int(frame_duration * samplerate))
                frame = np.squeeze(frame)
                amplitude = np.max(np.abs(frame))
                
                if amplitude > self.energy_threshold:
                    if not speech_started:
                        if debug:
                            print("🗣️  Speech detected, recording...")
                        speech_started = True
                    recording.append(frame)
                    silence_counter = 0
                    
                elif speech_started:
                    recording.append(frame)
                    silence_counter += frame_duration
                    
                    if silence_counter >= self.silence_duration:
                        if debug:
                            print("🔇 Silence detected, processing audio...")
                        break
        
        if not recording:
            return False, None
        
        # Convert recording to format for Whisper
        audio = np.concatenate(recording)
        
        # Resample to 16kHz for Whisper if needed
        if samplerate != 16000:
            audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)
        
        # Quick transcription with tiny Whisper model
        try:
            result = self.whisper_model.transcribe(audio, fp16=False)
            transcribed_text = result['text'].strip()
            
            if debug:
                print(f"Transcribed: '{transcribed_text}'")
            
            # Check if wake phrase is in transcribed text
            wake_detected = self.wake_phrase in transcribed_text.lower()
            
            if wake_detected:
                print(f"🎯 Wake phrase '{self.wake_phrase}' detected!")
            
            return wake_detected, transcribed_text
            
        except Exception as e:
            print(f"❌ Error in wake word verification: {e}")
            return False, None

def create_wake_word_detector(method: str = "ast", **kwargs) -> object:
    """
    Factory function to create wake word detector.
    
    Args:
        method: "ast" for MIT AST model, "custom" for energy+whisper approach
        **kwargs: Additional arguments passed to detector constructor
    """
    if method == "ast" or method == "classification":
        # Use MIT AST model for efficient wake word detection
        return ASTWakeWordDetector(
            wake_word="seven",  # Available in Speech Commands dataset
            confidence_threshold=kwargs.get("confidence_threshold", 0.7),
            **kwargs
        )
    elif method == "custom":
        # Use custom energy detection + Whisper verification
        return CustomWakeWordDetector(
            wake_phrase=kwargs.get("wake_phrase", BOT_NAME),
            energy_threshold=kwargs.get("energy_threshold", 0.02),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ast' or 'custom'")

# Test functions for debugging
def test_ast_detector(audio_device_index: int = None):
    """Test the AST-based wake word detector"""
    if audio_device_index is None:
        # Try to find a working audio device
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                audio_device_index = i
                break
        if audio_device_index is None:
            print("❌ No input audio device found")
            return
    
    detector = create_wake_word_detector("ast")
    
    print("Say 'seven' to test the AST wake word detector...")
    print("Press Ctrl+C to stop")
    
    detected = detector.listen_for_wake_word(audio_device_index, debug=True)
    if detected:
        print("✅ Wake word detection successful!")
    else:
        print("❌ Wake word detection failed or interrupted")

def test_custom_detector(audio_device_index: int = None):
    """Test the custom wake word detector"""
    if audio_device_index is None:
        # Try to find a working audio device
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                audio_device_index = i
                break
        if audio_device_index is None:
            print("❌ No input audio device found")
            return
    
    detector = create_wake_word_detector("custom")
    
    print(f"Say '{BOT_NAME}' to test the custom wake word detector...")
    print("Press Ctrl+C to stop")
    
    try:
        detected, text = detector.listen_for_wake_word(audio_device_index, debug=True)
        if detected:
            print(f"✅ Wake word detection successful! Heard: '{text}'")
        else:
            print(f"❌ Wake word not detected. Heard: '{text}'")
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")

if __name__ == "__main__":
    # Simple test script
    print("Wake Word Detector Test")
    print("1. AST method (say 'seven')")
    print("2. Custom method (say 'Overlord')")
    
    choice = input("Choose method (1 or 2): ").strip()
    
    if choice == "1":
        test_ast_detector()
    elif choice == "2":
        test_custom_detector()
    else:
        print("Invalid choice") 