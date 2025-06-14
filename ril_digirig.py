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
from scipy.signal import resample

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
        print(f"RadioInterfaceLayerDigiRig initialized. Device: '{self.device_name}', Samplerate: {self.samplerate}, Channels: {self.channels}")

    def _find_digirig_device(self):
        """
        Finds the Digirig audio device by name and sets its properties.
        """
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if 'USB PnP Sound Device' in device['name']:
                if device.get('max_input_channels', 0) > 0 and device.get('max_output_channels', 0) > 0:
                    print(f"Found Digirig device at index {i}: {device['name']}")
                    self.audio_device_index = i
                    self.device_name = device['name']
                    self.samplerate = int(device.get('default_samplerate', 44100))
                    self.channels = 1
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
            self.serial_conn.setRTS(False)
            self.serial_conn.setDTR(False)
        except serial.SerialException as e:
            print(f"[ERROR] Failed to open serial port {port_name} for Digirig: {e}")
            self.serial_conn = None
            raise RuntimeError(f"Failed to initialize PTT on {port_name}. Radio control will not be possible.") from e

    def check_carrier_sense(self, duration=CARRIER_SENSE_DURATION, threshold=AUDIO_THRESHOLD):
        """
        Checks for audio input (carrier) on the radio frequency.
        Returns True if carrier is detected, False if frequency is clear.
        """
        if self.audio_device_index is None:
            print("⚠️ Cannot check carrier sense: Digirig audio device not configured.")
            return False
            
        try:
            with sd.InputStream(device=self.audio_device_index, channels=1, samplerate=self.samplerate) as stream:
                time.sleep(0.1)
                audio_chunk, overflowed = stream.read(int(self.samplerate * duration))
                
                if overflowed:
                    print("⚠️ Input overflow while checking for carrier sense!")

                amplitude = np.sqrt(np.mean(np.square(audio_chunk)))

                if amplitude > threshold:
                    print(f"📡 Carrier detected (amplitude: {amplitude:.4f})")
                    return True
        except Exception as e:
            print(f"❌ Could not check carrier sense on device {self.audio_device_index}: {e}")
            return False
            
        return False

    def ptt_on(self, max_retries=CARRIER_SENSE_MAX_RETRIES, retry_delay=CARRIER_SENSE_RETRY_DELAY):
        """
        Activates PTT via serial port with carrier sense.
        """
        if not (self.serial_conn and self.serial_conn.is_open):
            print("Warning: Serial connection not available or not open for PTT ON.")
            return
            
        for attempt in range(max_retries):
            if not self.check_carrier_sense():
                try:
                    self.serial_conn.setRTS(True)
                    time.sleep(0.1)
                    print("\033[92m📡 PTT ON\033[0m")
                    return
                except serial.SerialException as e:
                    print(f"Error setting PTT ON: {e}")
                    return
            else:
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
                time.sleep(0.1)
                self.serial_conn.setRTS(False)
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
            sd.stop()
            time.sleep(0.1)
            sd.query_devices(self.audio_device_index)
            print(f"🎤 Digirig audio device {self.audio_device_index} ({self.device_name}) reset.")
            return True
        except Exception as e:
            print(f"⚠️ Digirig audio device reset warning for device {self.audio_device_index}: {e}")
            return False

    def play_audio(self, audio_data, sample_rate):
        """
        Plays the given audio data through the Digirig device.
        """
        if self.audio_device_index is None:
            print("Error: Digirig audio device not configured for playback.")
            return

        if sample_rate != self.samplerate:
            print(f"🔄 Resampling audio from {sample_rate} Hz to {self.samplerate} Hz...")
            try:
                new_length = int(len(audio_data) * self.samplerate / sample_rate)
                audio_data = resample(audio_data, new_length)
            except Exception as e:
                print(f"❌ Error during resampling: {e}")
                return
        
        try:
            print(f"🔊 Playing audio via Digirig on device {self.audio_device_index} at {self.samplerate} Hz...")
            sd.play(audio_data, samplerate=self.samplerate, device=self.audio_device_index, blocking=True)
        except Exception as e:
            print(f"Error during Digirig audio playback: {e}")

    def get_input_stream_params(self):
        """Returns parameters for sd.InputStream."""
        return {'device': self.audio_device_index, 'samplerate': self.samplerate, 'channels': self.channels, 'dtype': 'float32'}
        
    def get_output_stream_params(self):
        """Returns parameters for sd.OutputStream."""
        return {'device': self.audio_device_index, 'samplerate': self.samplerate, 'channels': self.channels, 'dtype': 'float32'}
        
    def get_samplerate(self):
        return self.samplerate

    def get_channels(self):
        return self.channels

    def get_audio_device_index(self):
        return self.audio_device_index

    def get_device_name(self):
        return self.device_name

    def close(self):
        """Closes the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.ptt_off()
            self.serial_conn.close()
            print("Digirig serial connection closed.")

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