# Changelog

All notable changes to the W6RGC-AI Ham Radio Voice Assistant project will be documented in this file.

## [2.2.0] - 2025-05-27

### Removed - Simplified Wake Word Detection

#### Wake Word System Simplification
- **Removed Custom Wake Word Detection**: Eliminated the Whisper-based custom wake word detector
  - Removed `CustomWakeWordDetector` class and related code
  - Removed "Overlord" custom wake phrase functionality
  - Simplified codebase by focusing on the more efficient AST method
  - Reduced dependencies and complexity

#### Code Cleanup
- **Streamlined `wake_word_detector.py`**: Now only contains AST-based detection
- **Simplified `main.py`**: Removed complex wake word method selection logic
- **Updated `constants.py`**: Removed custom wake word related constants
- **Cleaner Architecture**: Single, well-tested wake word detection method

### Changed
- **Wake Word Detection**: Now exclusively uses AST method with "seven" as the default wake word
- **Improved Performance**: Reduced memory usage and startup time by removing Whisper dependency for wake words
- **Better Reliability**: Single detection method reduces potential points of failure
- **Simplified Configuration**: Fewer options to configure, easier to set up and maintain

### Technical Improvements
- **Reduced Complexity**: Eliminated dual detection system complexity
- **Better Maintainability**: Single code path for wake word detection
- **Improved Documentation**: Clearer documentation focusing on supported features
- **Faster Startup**: No need to load additional Whisper models for wake word detection

### Added - Command Detection Refinement
- **`MAX_COMMAND_WORDS` Constant**: Introduced `MAX_COMMAND_WORDS` in `constants.py` to limit the number of words checked for voice commands. This prevents accidental command triggers during extended speech by only analyzing the beginning of an utterance for commands. Updated `commands.py` to use this constant.

## [2.1.0] - 2025-01-XX

### Added - Code Organization and Maintainability Improvements

#### Constants Centralization
- **`constants.py`**: Created comprehensive constants file with organized sections:
  - Radio/Ham Radio configuration (bot identity, callsigns)
  - Audio processing constants (thresholds, sample rates, channels)
  - Wake word detection settings (AST method)
  - Hardware configuration (serial ports, timeouts)
  - AI/LLM configuration (Ollama URL, models)
  - TTS configuration (models, settings, file paths)
  - Testing/debug constants (test frequencies, durations, text)
  - Device detection settings (CUDA/CPU, file paths)
  - Performance tuning parameters

#### Radio Interface Layer (RIL) Refactoring
- **`ril_aioc.py`**: Separated hardware interface logic into dedicated module
  - AIOC device detection and configuration
  - Serial PTT control management
  - Audio device parameter management
  - Hardware reset and error handling
  - Improved separation of concerns between TTS and hardware control

#### Enhanced Code Structure
- **Class-based PromptManager**: Refactored prompts.py into proper class structure
  - Better encapsulation of prompt management
  - Improved context handling
  - Support for multiple prompt types (original, radio_script)
  - Enhanced testing capabilities

### Changed
- **All modules updated** to import from centralized `constants.py`:
  - `prompts.py`: Now uses constants for bot identity
  - `ril_aioc.py`: Uses hardware and audio constants
  - `wake_word_detector.py`: Uses wake word detection constants
  - `main.py`: Uses all configuration constants, removed redundant definitions
  - `test_tts_performance.py`: Uses test-related constants
- **Improved maintainability**: All hardcoded values centralized for easy modification
- **Better documentation**: Each constant clearly documented with purpose and usage

### Technical Improvements
- **Configuration Management**: Single source of truth for all application settings
- **Code Reusability**: Constants can be easily shared across modules
- **Environment Flexibility**: Easy to modify settings for different deployments
- **Type Safety**: Constants clearly defined with meaningful names
- **Reduced Duplication**: Eliminated redundant hardcoded values throughout codebase

### Developer Experience
- **Easier Configuration**: Change settings in one place affects entire application
- **Better Testing**: Centralized test constants for consistent testing
- **Cleaner Code**: Removed magic numbers and hardcoded strings
- **Improved Readability**: Self-documenting constant names

## [2.0.0] - 2025-01-XX

### Added - Major Wake Word Detection Overhaul

#### AST Wake Word Detection System
- **AST Method**: Implemented MIT/ast-finetuned-speech-commands-v2 model for efficient wake word detection
  - 35+ pre-trained wake words available: backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero
  - Very fast, low CPU usage, high accuracy
  - Current default: "seven"
  - CUDA acceleration support

#### Hardware Integration Improvements
- **Automatic AIOC Detection**: System now automatically finds and configures All-In-One-Cable adapters
- **Sample Rate Conversion**: Automatic conversion between device rates (48kHz) and model requirements (16kHz) using librosa
- **Serial Port Auto-detection**: Improved serial port handling with better error messages
- **Audio Device Management**: Robust audio device detection with fallback options

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
- **Wake Word Processing**: Implemented efficient AST-based wake word detection system
- **Audio Pipeline**: Enhanced audio processing with proper sample rate handling
- **Bot Name Handling**: Flexible bot name recognition
- **Command Processing**: Streamlined command flow after wake word detection
- **Documentation**: Comprehensive updates to README.md and code comments

### Technical Improvements
- **Performance**: AST method provides fast, efficient wake word detection
- **Accuracy**: Improved wake word detection accuracy with confidence thresholds
- **Reliability**: Better error handling and recovery mechanisms
- **Simplicity**: Single, well-tested detection method

### Configuration Updates
- **main.py**: Updated to support AST detection method
- **wake_word_detector.py**: Focused module with AST detection capabilities
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
