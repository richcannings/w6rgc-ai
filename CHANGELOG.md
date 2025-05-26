# Changelog

All notable changes to the W6RGC-AI Ham Radio Voice Assistant project will be documented in this file.

## [2.0.0] - 2025-01-XX

### Added - Major Wake Word Detection Overhaul

#### Dual Wake Word Detection System
- **AST Method**: Implemented MIT/ast-finetuned-speech-commands-v2 model for efficient wake word detection
  - 35+ pre-trained wake words available: backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero
  - Very fast, low CPU usage, high accuracy
  - Current default: "seven"
  - CUDA acceleration support

- **Custom Method**: Enhanced Whisper-based detection for flexible wake phrases
  - Energy detection + Whisper verification
  - Can use any custom phrase (current: "Overlord")
  - Higher CPU usage but maximum flexibility
  - Full transcription capabilities

#### Hardware Integration Improvements
- **Automatic AIOC Detection**: System now automatically finds and configures All-In-One-Cable adapters
- **Sample Rate Conversion**: Automatic conversion between device rates (48kHz) and model requirements (16kHz) using librosa
- **Serial Port Auto-detection**: Improved serial port handling with better error messages
- **Audio Device Management**: Robust audio device detection with fallback options

#### New Testing Infrastructure
- **test_wake_word.py**: Comprehensive test suite for both detection methods
  - Interactive menu system
  - Real-time debug output
  - Confidence score monitoring
  - Hardware compatibility testing

#### Enhanced User Experience
- **Improved Logging**: Better status messages with emoji indicators
- **Debug Mode**: Detailed prediction output for tuning and troubleshooting
- **Graceful Error Handling**: Better error messages and recovery
- **Multiple Termination Methods**: Voice commands (break/exit/quit/shutdown) and Ctrl+C

#### Dependencies Added
- `transformers`: For AST model support
- `torch`: For CUDA acceleration
- `librosa`: For audio sample rate conversion

### Changed
- **Wake Word Processing**: Completely rewritten wake word detection system
- **Audio Pipeline**: Enhanced audio processing with proper sample rate handling
- **Bot Name Handling**: More flexible bot name recognition (accepts "7", "seven", "overlord")
- **Command Processing**: Streamlined command flow after wake word detection
- **Documentation**: Comprehensive updates to README.md and code comments

### Technical Improvements
- **Performance**: AST method provides ~10x faster wake word detection
- **Accuracy**: Improved wake word detection accuracy with confidence thresholds
- **Reliability**: Better error handling and recovery mechanisms
- **Scalability**: Modular design allows easy addition of new detection methods

### Configuration Updates
- **main.py**: Updated to support dual detection methods
- **wake_word_detector.py**: New module with comprehensive detection capabilities
- **prompts.py**: Enhanced bot configuration options
- **requirements.txt**: Updated with new dependencies

## [1.0.0] - 2025-01-XX

### Initial Release
- Basic voice assistant functionality
- Whisper speech recognition
- Ollama LLM integration
- CoquiTTS text-to-speech
- AIOC adapter support
- Serial PTT control
- Basic wake word detection 
