#!/usr/bin/env python3
"""
Audio Level Monitor for W6RGC/AI Ham Radio Voice Assistant

This module provides real-time audio level monitoring and analysis for optimizing
speech recognition performance. It tracks peak and RMS levels and provides
recommendations for audio input gain adjustments.

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
import time
from collections import deque


class AudioLevelMonitor:
    """
    Monitor and analyze audio input levels for optimization recommendations.
    
    This class provides real-time analysis of audio levels including peak and RMS
    calculations, automatic reporting, and optimization recommendations for
    speech recognition performance.
    """
    
    def __init__(self, window_size=50):
        """
        Initialize the audio level monitor.
        
        Args:
            window_size (int): Number of frames to keep in the analysis window
        """
        self.window_size = window_size
        self.peak_levels = deque(maxlen=window_size)
        self.rms_levels = deque(maxlen=window_size)
        self.frame_count = 0
        self.last_report_time = time.time()
        self.report_interval = 5.0  # Report every 5 seconds
        
        # Optimal levels for speech recognition
        self.optimal_peak_min = 0.1    # Minimum peak for good SNR
        self.optimal_peak_max = 0.8    # Maximum peak to avoid clipping
        self.optimal_rms_min = 0.05    # Minimum RMS for speech
        self.optimal_rms_max = 0.3     # Maximum RMS for clear speech
        
    def add_frame(self, audio_frame):
        """
        Add an audio frame for analysis.
        
        Args:
            audio_frame (np.ndarray): Audio frame data for analysis
        """
        self.frame_count += 1
        
        # Calculate peak and RMS for this frame
        peak = np.max(np.abs(audio_frame))
        rms = np.sqrt(np.mean(audio_frame**2))
        
        self.peak_levels.append(peak)
        self.rms_levels.append(rms)
        
        # Periodic reporting
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self._report_levels()
            self.last_report_time = current_time
    
    def _report_levels(self):
        """Generate audio level report and recommendations."""
        if not self.peak_levels or not self.rms_levels:
            return
            
        # Calculate statistics
        avg_peak = np.mean(self.peak_levels)
        max_peak = np.max(self.peak_levels)
        avg_rms = np.mean(self.rms_levels)
        max_rms = np.max(self.rms_levels)
        
        # Generate report
        print(f"🔊 Audio Levels Report (last {len(self.peak_levels)} frames):")
        print(f"   Peak: avg={avg_peak:.3f}, max={max_peak:.3f}")
        print(f"   RMS:  avg={avg_rms:.3f}, max={max_rms:.3f}")
        
        # Generate recommendations
        recommendations = []
        
        if avg_peak < self.optimal_peak_min:
            recommendations.append(f"📈 INCREASE input gain - Peak too low ({avg_peak:.3f} < {self.optimal_peak_min})")
        elif avg_peak > self.optimal_peak_max:
            recommendations.append(f"📉 DECREASE input gain - Peak too high ({avg_peak:.3f} > {self.optimal_peak_max})")
        
        if avg_rms < self.optimal_rms_min:
            recommendations.append(f"📈 INCREASE input gain - RMS too low ({avg_rms:.3f} < {self.optimal_rms_min})")
        elif avg_rms > self.optimal_rms_max:
            recommendations.append(f"📉 DECREASE input gain - RMS too high ({avg_rms:.3f} > {self.optimal_rms_max})")
        
        if max_peak > 0.95:
            recommendations.append("⚠️  CLIPPING DETECTED - Significantly reduce input gain!")
        
        if not recommendations:
            print("✅ Audio levels are optimal for speech recognition")
        else:
            for rec in recommendations:
                print(f"   {rec}")
        
        # Provide specific guidance
        if avg_peak < 0.05:
            print("💡 Try: Increase mic gain, move closer to microphone, or speak louder")
        elif avg_peak > 0.9:
            print("💡 Try: Decrease mic gain, move away from microphone, or speak softer")
        
        print(f"🎯 Target ranges: Peak {self.optimal_peak_min}-{self.optimal_peak_max}, RMS {self.optimal_rms_min}-{self.optimal_rms_max}")
        print()
    
    def get_current_stats(self):
        """
        Get current audio statistics.
        
        Returns:
            dict or None: Dictionary containing current audio statistics,
                         or None if no data is available
        """
        if not self.peak_levels or not self.rms_levels:
            return None
            
        return {
            'avg_peak': np.mean(self.peak_levels),
            'max_peak': np.max(self.peak_levels),
            'avg_rms': np.mean(self.rms_levels),
            'max_rms': np.max(self.rms_levels),
            'frame_count': self.frame_count
        } 