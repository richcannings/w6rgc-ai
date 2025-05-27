import sounddevice as sd
import serial
import time
import numpy as np
from constants import (
    DEFAULT_SERIAL_PORT,
    SERIAL_TIMEOUT,
    DEFAULT_AUDIO_CHANNELS,
    WHISPER_TARGET_SAMPLE_RATE,
    TEST_FREQUENCY,
    TEST_DURATION,
    TEST_AMPLITUDE
)

class RadioInterfaceLayerAIOC:
    def __init__(self, serial_port_name=DEFAULT_SERIAL_PORT):
        """
        Initializes the Radio Interface Layer for the AIOC.
        Detects the AIOC audio device and sets up PTT control.
        """
        self.audio_device_index = None
        self.samplerate = None
        self.channels = None
        self.serial_conn = None
        self.device_name = None

        self._find_aioc_device()
        self._setup_serial(serial_port_name)
        print(f"RadioInterfaceLayerAIOC initialized. Device: {self.device_name}, Samplerate: {self.samplerate}, Channels: {self.channels}")

    def _find_aioc_device(self):
        """
        Finds the AIOC audio device index, name, samplerate, and channels.
        Raises RuntimeError if not found.
        """
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if "All-In-One-Cable" in device.get('name', ''):
                if device.get('max_input_channels', 0) > 0 and device.get('max_output_channels', 0) > 0:
                    print(f"Found AIOC device at index {i}: {device['name']}")
                    self.audio_device_index = i
                    self.device_name = device['name']
                    self.samplerate = int(device.get('default_samplerate', WHISPER_TARGET_SAMPLE_RATE)) # Default to 16k if not specified
                    
                    # Determine input channels
                    # Prefer 'max_input_channels' but ensure it's at least 1 if device is valid input
                    # Some devices might report preferred channels as 0, so check max_input_channels
                    self.channels = device.get('max_input_channels', 0)
                    if self.channels == 0 and device.get('max_input_channels',0) > 0 : # Corrected this logic
                        self.channels = DEFAULT_AUDIO_CHANNELS # Default to mono if max_input_channels > 0
                    elif self.channels == 0 : # if still 0, means no input channels
                         print(f"Warning: AIOC device {device['name']} reports 0 input channels despite max_input_channels > 0. Defaulting to 1 channel.")
                         self.channels = DEFAULT_AUDIO_CHANNELS # Fallback if device has input capability but reports 0 channels.

                    if self.channels == 0: # Final check
                        raise RuntimeError(f"AIOC device {device['name']} has no usable input channels.")

                    print(f"AIOC Config: Index={self.audio_device_index}, Name='{self.device_name}', SampleRate={self.samplerate}, InputChannels={self.channels}")
                    return
        
        print("AIOC device not found. Available audio devices:")
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device['name']} (in:{device['max_input_channels']}, out:{device['max_output_channels']})")
        raise RuntimeError("AIOC (All-In-One-Cable) device not found. Please check your USB connection.")

    def _setup_serial(self, port_name):
        """
        Sets up the serial connection for PTT.
        Raises Exception if connection fails.
        """
        try:
            self.serial_conn = serial.Serial(port_name, timeout=SERIAL_TIMEOUT)
            print(f"Serial port {port_name} opened successfully for AIOC PTT.")
            self.serial_conn.setRTS(False)  # PTT Off
            self.serial_conn.setDTR(False)  # PTT Off (some radios use DTR)
        except serial.SerialException as e:
            print(f"[ERROR] Failed to open serial port {port_name} for AIOC: {e}")
            self.serial_conn = None
            # Depending on requirements, you might want to allow operation without PTT
            # For now, re-raising to indicate a critical setup failure.
            raise RuntimeError(f"Failed to initialize PTT on {port_name}. Radio control will not be possible.") from e


    # TODO: Check that the frequency is not in use before activating PTT. If in use, sleep for 3 
    # seconds and try again.
    def ptt_on(self):
        """Activates PTT via serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.setRTS(True)
                self.serial_conn.setDTR(True)
                time.sleep(0.1)  # Delay for PTT activation
                print("PTT ON")
            except serial.SerialException as e:
                print(f"Error setting PTT ON: {e}")
        else:
            print("Warning: Serial connection not available or not open for PTT ON.")

    def ptt_off(self):
        """Deactivates PTT via serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                time.sleep(0.1) # Small delay before PTT off
                self.serial_conn.setRTS(False)
                self.serial_conn.setDTR(False)
                print("PTT OFF")
            except serial.SerialException as e:
                print(f"Error setting PTT OFF: {e}")
        else:
            print("Warning: Serial connection not available or not open for PTT OFF.")

    def reset_audio_device(self):
        """
        Resets the AIOC audio device by stopping current operations.
        """
        if self.audio_device_index is None:
            print("⚠️ AIOC audio device index not set, cannot reset.")
            return False
        try:
            sd.stop()  # Stop any ongoing playback or recording on any device
            time.sleep(0.1) # Give it a moment to settle
            # Verify the device is still responsive
            sd.query_devices(self.audio_device_index) 
            print(f"🎤 AIOC audio device {self.audio_device_index} ({self.device_name}) reset.")
            return True
        except Exception as e:
            print(f"⚠️  AIOC audio device reset warning for device {self.audio_device_index}: {e}")
            return False

    def play_audio(self, audio_data, sample_rate):
        """
        Plays the given audio data through the AIOC device.
        Assumes audio_data is already prepared (numpy array, mono, normalized).
        """
        if self.audio_device_index is None:
            print("Error: AIOC audio device not configured for playback.")
            return
        
        # Audio data preparation (numpy conversion, mono, normalization)
        # is now expected to be done by the caller in main.py.

        # self.reset_audio_device() # Caller (main.py) should handle reset if needed.
        # self.ptt_on() # Caller (main.py) should handle PTT.
        try:
            print(f"🔊 Playing audio via AIOC (raw) on device {self.audio_device_index} at {sample_rate} Hz...")
            sd.play(
                audio_data,
                samplerate=sample_rate,
                device=self.audio_device_index,
                blocking=True
            )
        except Exception as e:
            print(f"Error during AIOC audio playback (sd.play): {e}")
        # finally: # PTT control is now handled by the caller.
            # self.ptt_off()

    def get_input_stream_params(self):
        """
        Returns a dictionary of parameters needed to open an input stream
        from the AIOC device.
        """
        if self.audio_device_index is None or self.samplerate is None or self.channels is None:
             raise RuntimeError("AIOC device not initialized properly to get input stream parameters.")
        return {
            "device": self.audio_device_index,
            "samplerate": self.samplerate,
            "channels": self.channels,
            "dtype": 'float32'  # Standard float32 for processing
        }

    def get_samplerate(self):
        """Returns the default samplerate of the AIOC input device."""
        if self.samplerate is None:
            raise RuntimeError("Samplerate not determined. AIOC device might not be initialized.")
        return self.samplerate

    def get_channels(self):
        """Returns the number of input channels of the AIOC device."""
        if self.channels is None:
            raise RuntimeError("Number of channels not determined. AIOC device might not be initialized.")
        return self.channels
        
    def get_audio_device_index(self):
        """Returns the audio device index for the AIOC."""
        if self.audio_device_index is None:
            raise RuntimeError("Audio device index not determined. AIOC device might not be initialized.")
        return self.audio_device_index

    def close(self):
        """Closes the serial connection if open."""
        if self.serial_conn and self.serial_conn.is_open:
            print("Closing AIOC serial port...")
            self.ptt_off() # Ensure PTT is off before closing
            self.serial_conn.close()
            print("AIOC serial port closed.")
        else:
            print("AIOC serial port was not open or not initialized.")

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    try:
        print("Attempting to initialize RadioInterfaceLayerAIOC...")
        # You might need to change "/dev/ttyACM0" to your actual AIOC serial port
        # or implement a discovery mechanism.
        aioc_ril = RadioInterfaceLayerAIOC(serial_port_name=DEFAULT_SERIAL_PORT) 
        print("RadioInterfaceLayerAIOC initialized successfully.")
        
        print(f"Device Index: {aioc_ril.get_audio_device_index()}")
        print(f"Samplerate: {aioc_ril.get_samplerate()}")
        print(f"Channels: {aioc_ril.get_channels()}")
        stream_params = aioc_ril.get_input_stream_params()
        print(f"Input Stream Params: {stream_params}")

        # Test PTT
        print("Testing PTT ON...")
        aioc_ril.ptt_on()
        time.sleep(1)
        print("Testing PTT OFF...")
        aioc_ril.ptt_off()

        # Test Playback (requires a dummy audio signal)
        print("Testing playback (dummy sine wave)...")
        samplerate = aioc_ril.get_samplerate()
        duration = TEST_DURATION  # seconds
        frequency = TEST_FREQUENCY  # Hz (A4 note)
        t = np.linspace(0, duration, int(samplerate * duration), False)
        dummy_audio = TEST_AMPLITUDE * np.sin(2 * np.pi * frequency * t)
        
        aioc_ril.play_audio(dummy_audio, samplerate)
        print("Playback test complete.")

        aioc_ril.close()
        print("RadioInterfaceLayerAIOC closed.")

    except RuntimeError as e:
        print(f"Runtime Error during RIL initialization or test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 