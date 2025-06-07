#!/usr/bin/env python3
# ril_digirig.py - Radio Interface Layer for Digirig
#
# This module provides the Radio Interface Layer (RIL) for Digirig
# hardware. It handles PTT (Push-to-Talk) control via serial port and manages
# audio device detection, configuration, and playback for the Digirig adapter.
# It also includes carrier sense functionality to check for channel activity
# before transmitting.
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

import sounddevice as sd
import serial
import time
import numpy as np
from constants import (
    DEFAULT_DIGIRIG_SERIAL_PORT,
    SERIAL_TIMEOUT,
    DEFAULT_AUDIO_CHANNELS,
    WHISPER_TARGET_SAMPLE_RATE,
    TEST_FREQUENCY,
    TEST_DURATION,
    TEST_AMPLITUDE,
    AUDIO_THRESHOLD,
    CARRIER_SENSE_DURATION,
    CARRIER_SENSE_MAX_RETRIES,
    CARRIER_SENSE_RETRY_DELAY
)

class RadioInterfaceLayerDigiRig:
    def __init__(self, serial_port_name=DEFAULT_DIGIRIG_SERIAL_PORT):
        """
        Initializes the Radio Interface Layer for the Digirig.
        Detects the Digirig audio device and sets up PTT control.
        """
        self.audio_device_index = None
        self.samplerate = None
        self.channels = None
        self.serial_conn = None
        self.device_name = None

        self._find_digirig_device()
        self._setup_serial(serial_port_name)
        print(f"RadioInterfaceLayerDigiRig initialized. Device: {self.device_name}, Samplerate: {self.samplerate}, Channels: {self.channels}")

    def _find_digirig_device(self):
        """
        Finds the Digirig audio device index, name, samplerate, and channels.
        Raises RuntimeError if not found.
        """
        devices = sd.query_devices()
        
        # Common Digirig device name patterns
        digirig_patterns = [
            "Digirig",
            "USB PnP Sound Device"
        ]
        
        for i, device in enumerate(devices):
            device_name = device.get('name', '')
            
            # Check if device name matches any Digirig pattern
            is_digirig = any(pattern in device_name for pattern in digirig_patterns)
            
            if is_digirig:
                if device.get('max_input_channels', 0) > 0 and device.get('max_output_channels', 0) > 0:
                    print(f"Found Digirig device at index {i}: {device['name']}")
                    self.audio_device_index = i
                    self.device_name = device['name']
                    self.samplerate = int(device.get('default_samplerate', WHISPER_TARGET_SAMPLE_RATE)) # Default to 16k if not specified
                    
                    # Determine input channels
                    # Prefer 'max_input_channels' but ensure it's at least 1 if device is valid input
                    # Some devices might report preferred channels as 0, so check max_input_channels
                    self.channels = 1 # device.get('max_input_channels', 0)
                    if self.channels == 0 and device.get('max_input_channels',0) > 0 : # Corrected this logic
                        self.channels = DEFAULT_AUDIO_CHANNELS # Default to mono if max_input_channels > 0
                    elif self.channels == 0 : # if still 0, means no input channels
                         print(f"Warning: Digirig device {device['name']} reports 0 input channels despite max_input_channels > 0. Defaulting to 1 channel.")
                         self.channels = DEFAULT_AUDIO_CHANNELS # Fallback if device has input capability but reports 0 channels.

                    if self.channels == 0: # Final check
                        raise RuntimeError(f"Digirig device {device['name']} has no usable input channels.")

                    print(f"Digirig Config: Index={self.audio_device_index}, Name='{self.device_name}', SampleRate={self.samplerate}, InputChannels={self.channels}")
                    return
        
        print("Digirig device not found. Available audio devices:")
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device['name']} (in:{device['max_input_channels']}, out:{device['max_output_channels']})")
        raise RuntimeError("Digirig device not found. Please check your USB connection.")

    def _setup_serial(self, port_name):
        """
        Sets up the serial connection for PTT.
        Raises Exception if connection fails.
        """
        try:
            self.serial_conn = serial.Serial(port_name, timeout=SERIAL_TIMEOUT)
            print(f"Serial port {port_name} opened successfully for Digirig PTT.")
            self.serial_conn.setRTS(False)  # PTT Off
            self.serial_conn.setDTR(False)  # PTT Off (some radios use DTR)
        except serial.SerialException as e:
            print(f"[ERROR] Failed to open serial port {port_name} for Digirig: {e}")
            self.serial_conn = None
            # Depending on requirements, you might want to allow operation without PTT
            # For now, re-raising to indicate a critical setup failure.
            raise RuntimeError(f"Failed to initialize PTT on {port_name}. Radio control will not be possible.") from e

    def _check_carrier_sense(self, duration=CARRIER_SENSE_DURATION):
        """
        Checks for audio input (carrier) on the radio frequency.
        Returns True if carrier is detected, False if frequency is clear.
        
        Args:
            duration (float): Duration in seconds to monitor for carrier
        """
        if self.audio_device_index is None:
            print("⚠️ Cannot check carrier sense: Digirig audio device not configured.")
            return False
            
        try:
            print(f"📡 Checking carrier sense for {duration}s...")
            
            # Get stream parameters
            stream_params = self.get_input_stream_params()
            
            # Calculate number of frames to read
            frames_to_read = int(duration * self.samplerate)
            frame_size = int(0.1 * self.samplerate)  # 100ms frames
            
            with sd.InputStream(**stream_params) as stream:
                for _ in range(0, frames_to_read, frame_size):
                    frame, _ = stream.read(frame_size)
                    frame = np.squeeze(frame)
                    amplitude = np.max(np.abs(frame))
                    
                    if amplitude > AUDIO_THRESHOLD:
                        print(f"📡 Carrier detected (amplitude: {amplitude:.4f})")
                        return True
                        
            print("📡 Frequency clear")
            return False
            
        except Exception as e:
            print(f"❌ Error during carrier sense check: {e}")
            # If we can't check, assume frequency is clear to avoid blocking transmission
            return False

    def ptt_on(self, max_retries=CARRIER_SENSE_MAX_RETRIES, retry_delay=CARRIER_SENSE_RETRY_DELAY):
        """
        Activates PTT via serial port with carrier sense.
        Checks for carrier before transmitting and retries if frequency is busy.
        
        Args:
            max_retries (int): Maximum number of carrier sense attempts
            retry_delay (float): Delay in seconds between retries when carrier is detected
        """
        if not (self.serial_conn and self.serial_conn.is_open):
            print("Warning: Serial connection not available or not open for PTT ON.")
            return
            
        # Carrier sense loop
        for attempt in range(max_retries):
            if not self._check_carrier_sense():
                # Frequency is clear, proceed with PTT
                try:
                    self.serial_conn.setRTS(True)
                    # self.serial_conn.setDTR(True)
                    time.sleep(0.1)  # Delay for PTT activation
                    print("\033[92m📡 PTT ON\033[0m")
                    return
                except serial.SerialException as e:
                    print(f"Error setting PTT ON: {e}")
                    return
            else:
                # Carrier detected, wait and retry
                if attempt < max_retries - 1:
                    print(f"📡 Frequency busy, waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(retry_delay)
                else:
                    print(f"📡 Frequency still busy after {max_retries} attempts. Stopping transmission.")
                    return

    def ptt_off(self):
        """Deactivates PTT via serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                time.sleep(0.1) # Small delay before PTT off
                self.serial_conn.setRTS(False)
                # self.serial_conn.setDTR(False)
                print("\033[91m📻 PTT OFF\033[0m")
            except serial.SerialException as e:
                print(f"Error setting PTT OFF: {e}")
        else:
            print("Warning: Serial connection not available or not open for PTT OFF.")

    def reset_audio_device(self):
        """
        Resets the Digirig audio device by stopping current operations.
        """
        if self.audio_device_index is None:
            print("⚠️ Digirig audio device index not set, cannot reset.")
            return False
        try:
            sd.stop()  # Stop any ongoing playback or recording on any device
            time.sleep(0.1) # Give it a moment to settle
            # Verify the device is still responsive
            sd.query_devices(self.audio_device_index) 
            print(f"🎤 Digirig audio device {self.audio_device_index} ({self.device_name}) reset.")
            return True
        except Exception as e:
            print(f"⚠️  Digirig audio device reset warning for device {self.audio_device_index}: {e}")
            return False

    def play_audio(self, audio_data, sample_rate):
        """
        Plays the given audio data through the Digirig device.
        Assumes audio_data is already prepared (numpy array, mono, normalized).
        Automatically resamples to device-supported sample rate if needed.
        """
        if self.audio_device_index is None:
            print("Error: Digirig audio device not configured for playback.")
            return
        
        # Check if we need to resample
        target_sample_rate = sample_rate
        supported_rates = [44100, 48000]  # Known supported rates for most USB audio devices
        
        if sample_rate not in supported_rates:
            # Resample to 44100 Hz (most commonly supported)
            target_sample_rate = 44100
            print(f"🔄 Resampling audio from {sample_rate} Hz to {target_sample_rate} Hz...")
            
            # Calculate new length
            new_length = int(len(audio_data) * target_sample_rate / sample_rate)
            
            # Resample using scipy
            from scipy.signal import resample
            audio_data = resample(audio_data, new_length)
            
            print(f"✅ Resampled audio: {len(audio_data)} samples at {target_sample_rate} Hz")

        try:
            print(f"🔊 Playing audio via Digirig (raw) on device {self.audio_device_index} at {target_sample_rate} Hz...")
            sd.play(
                audio_data,
                samplerate=target_sample_rate,
                device=self.audio_device_index,
                blocking=True
            )
        except Exception as e:
            print(f"Error during Digirig audio playback (sd.play): {e}")
        # finally: # PTT control is now handled by the caller.
            # self.ptt_off()

    def get_input_stream_params(self):
        """
        Returns a dictionary of parameters needed to open an input stream
        from the Digirig device.
        """
        if self.audio_device_index is None or self.samplerate is None or self.channels is None:
             raise RuntimeError("Digirig device not initialized properly to get input stream parameters.")
        return {
            "device": self.audio_device_index,
            "samplerate": self.samplerate,
            "channels": self.channels,
            "dtype": 'float32'  # Standard float32 for processing
        }

    def get_samplerate(self):
        """Returns the default samplerate of the Digirig input device."""
        if self.samplerate is None:
            raise RuntimeError("Samplerate not determined. Digirig device might not be initialized.")
        return self.samplerate

    def get_channels(self):
        """Returns the number of input channels of the Digirig device."""
        if self.channels is None:
            raise RuntimeError("Number of channels not determined. Digirig device might not be initialized.")
        return self.channels
        
    def get_audio_device_index(self):
        """Returns the audio device index for the Digirig."""
        if self.audio_device_index is None:
            raise RuntimeError("Audio device index not determined. Digirig device might not be initialized.")
        return self.audio_device_index

    def close(self):
        """Closes the serial connection if open."""
        if self.serial_conn and self.serial_conn.is_open:
            print("Closing Digirig serial port...")
            self.ptt_off() # Ensure PTT is off before closing
            self.serial_conn.close()
            print("Digirig serial port closed.")
        else:
            print("Digirig serial port was not open or not initialized.")

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    try:
        print("Attempting to initialize RadioInterfaceLayerDigiRig...")
        # You might need to change "/dev/ttyACM0" to your actual Digirig serial port
        # or implement a discovery mechanism.
        digirig_ril = RadioInterfaceLayerDigiRig(serial_port_name=DEFAULT_DIGIRIG_SERIAL_PORT) 
        print("RadioInterfaceLayerDigiRig initialized successfully.")
        
        print(f"Device Index: {digirig_ril.get_audio_device_index()}")
        print(f"Samplerate: {digirig_ril.get_samplerate()}")
        print(f"Channels: {digirig_ril.get_channels()}")
        stream_params = digirig_ril.get_input_stream_params()
        print(f"Input Stream Params: {stream_params}")

        # Test PTT
        print("\033[94m🧪 Testing PTT ON...\033[0m")
        digirig_ril.ptt_on()
        time.sleep(1)
        print("\033[94m🧪 Testing PTT OFF...\033[0m")
        digirig_ril.ptt_off()

        # Test Playback (requires a dummy audio signal)
        print("Testing playback (dummy sine wave)...")
        samplerate = digirig_ril.get_samplerate()
        duration = TEST_DURATION  # seconds
        frequency = TEST_FREQUENCY  # Hz (A4 note)
        t = np.linspace(0, duration, int(samplerate * duration), False)
        dummy_audio = TEST_AMPLITUDE * np.sin(2 * np.pi * frequency * t)
        
        digirig_ril.play_audio(dummy_audio, samplerate)
        print("Playback test complete.")

        digirig_ril.close()
        print("RadioInterfaceLayerDigiRig closed.")

    except RuntimeError as e:
        print(f"Runtime Error during RIL initialization or test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 