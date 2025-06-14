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
    DEFAULT_SAMPLE_RATE = 16000  # Default to 16kHz for Whisper/AST compatibility
    
    def __init__(self, serial_port_name=DEFAULT_DIGIRIG_SERIAL_PORT, sample_rate=DEFAULT_SAMPLE_RATE, input_channels=1):
        """
        Initializes the Digirig Radio Interface Layer.

        Args:
            serial_port_name (str): The serial port for PTT control.
            sample_rate (int): The sample rate for the audio device.
            input_channels (int): The number of input channels.
        """
        print(f"🔧 Initializing Digirig Radio Interface Layer...")
        
        # Audio device configuration
        self.audio_device_index = self._find_digirig_device()
        self.samplerate = sample_rate
        self.input_channels = input_channels
        self.serial_conn = None
        self.device_name = None
        self.ptt_serial = None
        
        if self.audio_device_index is None:
            raise RuntimeError("Could not find a Digirig audio device.")
            
        print(f"Digirig Config: Index={self.audio_device_index}, Name='{self.get_device_name()}', SampleRate={self.samplerate}, InputChannels={self.input_channels}")

        # Initialize PTT control
        try:
            self._setup_serial(serial_port_name)
        except Exception as e:
            raise RuntimeError(f"Failed to open Digirig serial port at {serial_port_name}: {e}")
            
        print(f"RadioInterfaceLayerDigiRig initialized. Device: {self.get_device_name()}, Samplerate: {self.samplerate}, Channels: {self.input_channels}")

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
                    return i
        
        print("Digirig device not found. Available audio devices:")
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device['name']} (in:{device['max_input_channels']}, out:{device['max_output_channels']})")
        return None

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

    def check_carrier_sense(self, duration=0.2, threshold=0.01):
        """
        Checks for audio input (carrier) on the radio frequency.
        Returns True if carrier is detected, False if frequency is clear.
        
        Args:
            duration (float): Duration in seconds to monitor for carrier
            threshold (float): Threshold amplitude to consider carrier detected
        """
        if self.audio_device_index is None:
            print("⚠️ Cannot check carrier sense: Digirig audio device not configured.")
            return False
            
        try:
            with sd.InputStream(device=self.audio_device_index, channels=1, samplerate=self.samplerate) as stream:
                # Let the stream settle
                time.sleep(0.1)
                
                # Read a chunk of audio
                audio_chunk, overflowed = stream.read(int(self.samplerate * duration))
                
                if overflowed:
                    print("⚠️ Input overflow while checking for carrier sense!")

                # Calculate RMS amplitude
                amplitude = np.sqrt(np.mean(np.square(audio_chunk)))

                if amplitude > threshold:
                    print(f"📡 Carrier detected (amplitude: {amplitude:.4f})")
                    return True

        except Exception as e:
            print(f"❌ Could not check carrier sense on device {self.audio_device_index}: {e}")
            # Assume channel is clear if we can't check
            return False
            
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
            if not self.check_carrier_sense():
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
            print(f"🎤 Digirig audio device {self.audio_device_index} ({self.get_device_name()}) reset.")
            return True
        except Exception as e:
            print(f"⚠️  Digirig audio device reset warning for device {self.audio_device_index}: {e}")
            return False

    def play_audio(self, audio_data, sample_rate):
        """
        Plays the given audio data through the Digirig device.
        Assumes audio_data is already prepared (numpy array, mono, normalized).
        Automatically resamples to the device's sample rate if needed.
        """
        if self.audio_device_index is None:
            print("Error: Digirig audio device not configured for playback.")
            return

        # Resample if the source sample rate doesn't match the device rate
        if sample_rate != self.samplerate:
            print(f"🔄 Resampling audio from {sample_rate} Hz to {self.samplerate} Hz...")
            try:
                from scipy.signal import resample
                new_length = int(len(audio_data) * self.samplerate / sample_rate)
                audio_data = resample(audio_data, new_length)
                print(f"✅ Resampled audio: {len(audio_data)} samples at {self.samplerate} Hz")
            except ImportError:
                print("⚠️ SciPy not installed. Cannot resample audio. Playback may fail or be distorted.")
                print("   Please install it with: pip install scipy")
            except Exception as e:
                print(f"❌ Error during resampling: {e}")
                return # Can't proceed if resampling fails
        
        try:
            print(f"🔊 Playing audio via Digirig (raw) on device {self.audio_device_index} at {self.samplerate} Hz...")
            sd.play(
                audio_data,
                samplerate=self.samplerate,
                device=self.audio_device_index,
                blocking=True
            )
        except Exception as e:
            print(f"Error during Digirig audio playback (sd.play): {e}")

    def get_input_stream_params(self):
        """Returns parameters for sd.InputStream."""
        return {
            'device': self.audio_device_index,
            'samplerate': self.samplerate,
            'channels': self.input_channels,
            'dtype': 'float32'
        }
        
    def get_output_stream_params(self):
        """Returns parameters for sd.OutputStream."""
        return {
            'device': self.audio_device_index,
            'samplerate': self.samplerate,
            'channels': self.get_channels(),
            'dtype': 'float32'
        }

    def get_samplerate(self):
        """Returns the default samplerate of the Digirig input device."""
        if self.samplerate is None:
            raise RuntimeError("Samplerate not determined. Digirig device might not be initialized.")
        return self.samplerate

    def get_channels(self):
        """Returns the number of input channels of the Digirig device."""
        if self.input_channels is None:
            raise RuntimeError("Number of channels not determined. Digirig device might not be initialized.")
        return self.input_channels
        
    def get_audio_device_index(self):
        """Returns the audio device index for the Digirig."""
        if self.audio_device_index is None:
            raise RuntimeError("Audio device index not determined. Digirig device might not be initialized.")
        return self.audio_device_index

    def get_device_name(self):
        """Returns the name of the Digirig audio device."""
        if self.device_name is None:
            raise RuntimeError("Device name not determined. Digirig device might not be initialized.")
        return self.device_name

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