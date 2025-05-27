#!/usr/bin/env python3
# wake_word_detector.py - AST-based wake word detection for ham radio AI assistant
#
# This module provides efficient wake word detection for the W6RGC/AI system using
# MIT's AST (Audio Spectrogram Transformer) model for fast, accurate detection.
#
# AST Method Features:
#    - Uses MIT/ast-finetuned-speech-commands-v2 model
#    - 35+ pre-trained wake words available
#    - Very fast, low CPU usage, high accuracy
#    - Current default: "seven"
#    - Available words: backward, bed, bird, cat, dog, down, eight, five, follow, 
#      forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, 
#      one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero
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
from constants import (
    DEFAULT_WAKE_WORD,
    AST_CONFIDENCE_THRESHOLD,
    AST_CHUNK_LENGTH_S,
    AST_MODEL_NAME,
    DEFAULT_DEVICE_SAMPLE_RATE,
    WHISPER_TARGET_SAMPLE_RATE,
    CUDA_DEVICE,
    CPU_DEVICE,
    WAKE_WORD_METHOD_AST,
    AUDIO_FRAME_MS
)

class ASTWakeWordDetector:
    """
    Efficient wake word detector using MIT/ast-finetuned-speech-commands-v2 model.
    This model is specifically trained on speech commands and is much more efficient
    than running Whisper continuously.
    """
    
    def __init__(self, 
                 wake_word: str = DEFAULT_WAKE_WORD,
                 confidence_threshold: float = AST_CONFIDENCE_THRESHOLD,
                 chunk_length_s: float = AST_CHUNK_LENGTH_S,
                 device_sample_rate: int = DEFAULT_DEVICE_SAMPLE_RATE):
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
        device = CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE
        print(f"Loading AST wake word detector on {device}...")
        
        self.classifier = pipeline(
            "audio-classification", 
            model=AST_MODEL_NAME, 
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

def create_wake_word_detector(method: str = WAKE_WORD_METHOD_AST, device_sample_rate: int = DEFAULT_DEVICE_SAMPLE_RATE, **kwargs) -> ASTWakeWordDetector:
    """
    Factory function to create a wake word detector instance.
    
    Args:
        method: Must be "ast" (only supported method)
        device_sample_rate: Sample rate of the audio input device.
        **kwargs: Additional arguments for the AST detector 
                  (e.g., wake_word, confidence_threshold).
    
    Returns:
        An instance of ASTWakeWordDetector.
    """
    if method.lower() == WAKE_WORD_METHOD_AST:
        # AST specific defaults if not provided in kwargs
        ast_wake_word = kwargs.get('wake_word', DEFAULT_WAKE_WORD)
        ast_confidence = kwargs.get('confidence_threshold', AST_CONFIDENCE_THRESHOLD)
        return ASTWakeWordDetector(
            wake_word=ast_wake_word,
            confidence_threshold=ast_confidence,
            device_sample_rate=device_sample_rate
        )
    else:
        raise ValueError(f"Unsupported wake word detection method: {method}. Only 'ast' is supported.")

# Test function for debugging
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
    
    detector = create_wake_word_detector(WAKE_WORD_METHOD_AST)
    
    print(f"Say '{DEFAULT_WAKE_WORD}' to test the AST wake word detector...")
    print("Press Ctrl+C to stop")
    
    detected = detector.listen_for_wake_word(audio_device_index, debug=True)
    if detected:
        print("✅ Wake word detection successful!")
    else:
        print("❌ Wake word detection failed or interrupted")

if __name__ == "__main__":
    # Simple test script
    print("AST Wake Word Detector Test")
    print(f"Say '{DEFAULT_WAKE_WORD}' to test the detector...")
    
    test_ast_detector() 