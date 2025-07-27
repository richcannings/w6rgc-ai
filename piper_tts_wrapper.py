#!/usr/bin/env python3
"""
Piper TTS Wrapper for W6RGC/AI Ham Radio Voice Assistant

This module provides a wrapper class for PiperTTS to make it API-compatible 
with CoquiTTS, allowing seamless switching between TTS engines.

Author: Rich Cannings <rcannings@gmail.com>
Copyright 2025 Rich Cannings

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from piper.voice import PiperVoice


class PiperTTSWrapper:
    """
    A wrapper for PiperTTS to make it API-compatible with CoquiTTS.
    
    This wrapper allows PiperTTS to be used as a drop-in replacement for CoquiTTS
    by providing the same interface methods and return formats.
    """
    
    def __init__(self, model_path):
        """
        Initialize the PiperTTS wrapper.
        
        Args:
            model_path (str): Path to the Piper .onnx model file
        """
        # Piper models are specified by a path to the .onnx file
        self.voice = PiperVoice.load(model_path)
        
        # Create a dummy synthesizer object to hold the sample rate
        # This mimics the CoquiTTS interface
        class Synthesizer:
            def __init__(self, sample_rate):
                self.output_sample_rate = sample_rate
                
        self.synthesizer = Synthesizer(self.voice.config.sample_rate)
        self.model_name = model_path

    def tts(self, text, **kwargs):
        """
        Synthesize text to speech and return as a numpy array.
        
        This method provides the same interface as CoquiTTS.tts() for compatibility.
        
        Args:
            text (str): Text to synthesize
            **kwargs: Additional arguments (ignored for Piper compatibility)
            
        Returns:
            np.ndarray: Audio data as a normalized float32 numpy array
        """
        # Use synthesize_stream_raw to get audio data in memory
        wav_bytes_stream = self.voice.synthesize_stream_raw(text)
        wav_bytes = b"".join(wav_bytes_stream)

        # Convert bytes to a numpy array of floats
        audio_array = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1] range to be compatible with Coqui output
        if np.max(np.abs(audio_array)) > 0:
            audio_array /= np.iinfo(np.int16).max
        
        return audio_array 