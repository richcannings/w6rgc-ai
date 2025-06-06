import whisper
import numpy as np
import sounddevice as sd
from scipy.signal import resample

from constants import (
    AUDIO_THRESHOLD,
    FRAME_DURATION,
    SILENCE_DURATION,
    WHISPER_TARGET_SAMPLE_RATE,
    WHISPER_MODEL
)

class SpeechRecognitionEngine:
    def __init__(self, ril_interface):
        self.model = whisper.load_model(WHISPER_MODEL)
        self.ril_interface = ril_interface
        
    @property
    def version(self):
        """Get the Whisper model version for status reporting."""
        return getattr(self.model, '_version', 'unknown')

    def transcribe(self, audio_data):
        """
        Transcribe audio data using the Whisper model.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            dict: Whisper transcription result
        """
        result = self.model.transcribe(audio_data, fp16=True, language='en')
        return result
    
    def get_full_command_after_wake_word(self):
        """
        Record and transcribe the full command after wake word detection.
        Uses the existing Whisper model for high-quality transcription.
        Uses RadioInterfaceLayerAIOC for audio input stream parameters.
        
        Returns:
            str: Transcribed text from the recorded audio
        """
        print("🎤 Recording your command...")
        
        # Reset audio device to ensure clean recording
        self.ril_interface.reset_audio_device()
        
        recording = []
        speech_started = False
        silence_counter = 0
        
        # Get stream parameters from RIL
        stream_params = self.ril_interface.get_input_stream_params()
        samplerate = stream_params['samplerate'] # Use RIL determined samplerate

        with sd.InputStream(**stream_params) as stream:
            while True:
                frame, _ = stream.read(int(FRAME_DURATION * samplerate))
                frame = np.squeeze(frame)
                amplitude = np.max(np.abs(frame))
                
                if amplitude > AUDIO_THRESHOLD:
                    if not speech_started:
                        print("🗣️  Speech detected, recording...")
                        speech_started = True
                    recording.append(frame)
                    silence_counter = 0
                elif speech_started:
                    recording.append(frame)
                    silence_counter += FRAME_DURATION
                    if silence_counter >= SILENCE_DURATION:
                        print("🔇 Silence detected, processing...")
                        break
        
        if not recording:
            print("❌ No speech detected")
            return ""
        
        # Process audio for Whisper
        audio = np.concatenate(recording)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if samplerate != WHISPER_TARGET_SAMPLE_RATE:
            num_samples = int(len(audio) * WHISPER_TARGET_SAMPLE_RATE / samplerate)
            audio = resample(audio, num_samples)
        
        # Transcribe with main Whisper model
        print("📝 Transcribing with Whisper...")
        result = self.model.transcribe(audio, fp16=True, language='en')
        transcribed_text = result['text'].strip()
        
        print(f"📝 Transcribed: '{transcribed_text}'")
        return transcribed_text