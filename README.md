# W6RGC/AI: Off grid ham radio AI voice assistant

## Overview

If "AI is the new electricity", then someday we'll need a backup AI buddy when communication systems go down. 

That's where W6RGC/AI comes in.

W6RGC/AI is a voice assistant for ham radio operators that runs entirely on your local computer. No internet? No problem!

It's a Python-powered helper that uses a sequence of four AI models to understand you and talk back:

1.  A smart wake word spotter (it listens for "seven" by default, but you can choose from over 35 options!)
2.  Speech-to-text using Whisper (so it understands what you say)
3.  A flexible large language model (LLM) for the brains of the operation (currently using Ollama's gemma3:12b). Plug in your own specialized models and prompts!
4.  Coqui text-to-speech to give it a voice

The app communicates with ham radios with either an [AIOC adapter](https://github.com/skuep/AIOC) or [Digirig](https://digirig.net/).

Just say the wake word (like "seven") and then tell it what you need. It's designed for easy, hands-free use in your radio shack or mobile

You can also use voice commands like:
- **"Status" or "Report"**: Say "seven, status" or "seven, report" to find out which AI model is active and the assistant's callsign.
- **"Reset" or "New Chat"**: Say "seven, reset" or "seven, start a new chat" to wipe the conversation slate clean.
- **"Identify"**: Say "seven, identify" or other phrases like "identify", "call sign", "what is your call sign", or "who are you" to hear the assistant's phonetic callsign.
- **"Terminate"**: Say "seven, break" or "seven, exit" to shut down the assistant.

## Purpose

This project is all about mixing the ever-relevant hobby of amateur radio with the new world of AI. The main idea is to see how AI can make ham radio even better, whether that's for critical communications during an emergency, or just for fun, everyday radio chats.

Think of it this way:
*   **When the internet's down (Offline):** This is your AI backup. Imagine an LLM trained with all sorts of useful info for ham radio operators – ARES manuals, FEMA documents, local emergency plans, and so on. It could be an extra "voice" helping you figure things out. Plus, things like voice translation could be added pretty easily.
*   **When you're online:** You can connect to even more powerful LLMs. Most online AIs can browse the internet and do all sorts of cool stuff.

One exciting idea for the future (and my next project!) is "Voice APRS." Imagine being able to:
* Send and receive APRS messages using just your voice.
* Beacon your location by saying something like, "This is W6RGC. I'm at the corner of West Cliff Drive and Swift Street in Santa Cruz. Beacon my position on APRS."

## Features

*   **Efficient Wake Word Detection:** Uses MIT's AST (Audio Spectrogram Transformer) model for fast, accurate detection
*   **Conversational AI:** Leverages Ollama with models like Gemma3 to understand and respond to user speech
*   **High-Quality Speech Recognition:** Utilizes OpenAI Whisper for accurate transcription of spoken audio
*   **Natural Text-to-Speech:** Employs CoquiTTS for generating spoken responses with CUDA acceleration
*   **Automatic Audio Device Detection:** Automatically finds and configures AIOC (All-In-One-Cable) adapters
*   **Sample Rate Conversion:** Handles audio conversion between device rates (48kHz) and model requirements (16kHz)
*   **Modular Architecture:** Clean separation of concerns with dedicated modules for hardware, prompts, and constants
*   **Centralized Configuration:** All settings managed through `constants.py` for easy customization
*   **Customizable Persona:** AI's name, callsign, and speaking style can be configured in `prompts.py`
*   **Contextual Conversation:** Maintains conversation history for more natural interactions
*   **PTT Control:** Integrates with serial PTT for transmitting AI responses over the air
*   **Carrier Sense:** Automatically checks for ongoing transmissions before keying PTT to avoid interference
*   **Periodic Identification:** Automatic station identification at configurable intervals
*   **Graceful Termination:** Recognizes voice commands like "break", "exit", "quit", or "shutdown" to gracefully shut down

## Architecture

The system is organized into several key modules:

- **`main.py`**: Main application entry point and orchestration
- **`constants.py`**: Centralized configuration management for all settings
- **`commands.py`**: Command identification and parsing for voice commands (status, reset, identify, terminate)
- **`ril_aioc.py`**: Radio Interface Layer for AIOC hardware management
- **`ril_digirig.py`**: Radio Interface Layer for Digirig hardware management
- **`prompts.py`**: AI persona and conversation management (class-based)
- **`wake_word_detector.py`**: AST-based wake word detection system
- **`periodically_identify.py`**: Handles periodic station identification

## Wake Word Detection

The system uses MIT's AST (Audio Spectrogram Transformer) model for efficient wake word detection:

- **35+ Available Wake Words:** backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero
- **Current Default:** "seven" (optimized for ham radio use)
- **High Performance:** Very fast, low CPU usage, high accuracy
- **CUDA Support:** GPU acceleration when available

To change the wake word, modify `DEFAULT_WAKE_WORD` in `constants.py` to any of the supported words.

## Technologies Used

*   **Python 3**
*   **Wake Word Detection:** MIT AST (Audio Spectrogram Transformer)
*   **Speech-to-Text:** [OpenAI Whisper](https://openai.com/research/whisper)
*   **Large Language Model Engine:** [Ollama](https://ollama.ai/) (e.g., Gemma3 models)
*   **Text-to-Speech:** [CoquiTTS](https://github.com/coqui-ai/TTS)
*   **Audio Processing:** `sounddevice`, `soundfile`, `numpy`, `scipy`, `librosa`
*   **Machine Learning:** `transformers`, `torch` (with CUDA support)
*   **Serial Communication:** `pyserial` (for PTT control)
*   **HTTP Requests:** `requests` (for Ollama API)

## Setup and Installation

### Prerequisites

*   **Operating System:** This project has primarily been tested on Linux
*   **Python:** Python 3.8 or higher
*   **CUDA (Optional):** For GPU acceleration of wake word detection and TTS
*   **Required Linux Packages:** Install the following packages using apt:
    ```bash
    sudo apt update
    sudo apt install python3-full portaudio19-dev ffmpeg
    ```
*   **User Permissions:** Add your user to the `dialout` group for serial port access:
    ```bash
    sudo usermod -a -G dialout $USER
    # Log out and back in for changes to take effect
    ```
*   **Ollama:** Install Ollama by running the following command in your terminal:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```    After installation, ensure the Ollama service is running and you have pulled a model (e.g., `ollama pull gemma3:12b`).
*   **AIOC (All-In-One-Cable) Adapter or Digirig:** This project is designed to work with an AIOC or Digirig adapter for PTT (Push-to-Talk) functionality. The system automatically detects AIOC and Digirig devices by name.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd w6rgc-ai
    ```
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up Ollama (if not done in prerequisites):**
    *   Ensure the Ollama server is running (usually `ollama serve` or `systemctl start ollama`)
    *   If you haven't already, pull your desired LLM model (e.g., `ollama pull gemma3:12b`)
5.  **Hardware Setup:**
    *   Connect your AIOC adapter via USB
    *   The system will automatically detect the audio device and serial port

## Usage

1.  **Configure the application:**
    *   **Primary Configuration**: Edit `constants.py` to modify all application settings:
      - Audio processing parameters (thresholds, sample rates)
      - Wake word selection (choose from 35+ available words)
      - Hardware configuration (RIL type selection: AIOC or Digirig, serial ports)
      - AI/LLM settings (Ollama URL, model selection)
      - TTS configuration (models, audio settings)
      - Bot identity (name, callsign, persona)
      - Periodic identification interval
    *   **Advanced Configuration**: Customize AI persona and prompts in `prompts.py`
    *   The system automatically detects AIOC/Digirig hardware, but you can manually set serial port in `constants.py`

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Operation:**
    *   The application will start listening for the wake word
    *   Say "seven" (or your configured wake word) followed by your command
    *   Example: "seven, what is the current UTC time?"
    *   To terminate: "seven, break" or "seven, exit" or use Ctrl+C

## Configuration

The application uses a centralized configuration system through `constants.py`. Key configuration sections include:

### AI/LLM Configuration
```python
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:12b"    # Ollama model to use
```

### Wake Word Detection
```python
DEFAULT_WAKE_WORD = BOT_NAME    # When the wake word and bots name is the same, the dialogue is more flowy. Limited choices though.
MAX_COMMAND_WORDS = 10          # Max words to check for commands (prevents accidental triggers)
```

### Hardware Configuration
```python
DEFAULT_RIL_TYPE = "digirig"  # Options: "aioc" or "digirig"
DEFAULT_AIOC_SERIAL_PORT = "/dev/ttyACM0"  # Serial port for AIOC PTT control
DEFAULT_DIGIRIG_SERIAL_PORT = "/dev/ttyUSB2" # Serial port for Digirig PTT control
```

### Bot Identity (also configurable in prompts.py)
```python
BOT_NAME = "7"                  # Bot's name in conversation
BOT_CALLSIGN = "W6RGC/AI"      # Amateur radio callsign
BOT_PHONETIC_CALLSIGN = "Whiskey 6 Radio Golf Charlie Stroke Artificial Intelligence"
```

### TTS Configuration
```python
TTS_MODEL_FAST_PITCH = "tts_models/en/ljspeech/fast_pitch"     # Fastest TTS model
TTS_MODEL_SPEEDY_SPEECH = "tts_models/en/ljspeech/speedy_speech" # Alternative fast model
TTS_MODEL_TACOTRON2 = "tts_models/en/ljspeech/tacotron2-DDC"   # Fallback model
```

### Carrier Sense Configuration
```python
CARRIER_SENSE_DURATION = 0.5      # seconds to monitor for carrier before PTT
CARRIER_SENSE_MAX_RETRIES = 3     # maximum attempts to find clear frequency  
CARRIER_SENSE_RETRY_DELAY = 3.0   # seconds to wait between carrier sense attempts
```

### Periodic Identification Configuration
```python
PERIODIC_ID_INTERVAL_MINUTES = 10  # minutes between automatic identification
```

## Hardware Requirements

*   **Minimum:** CPU with 4+ cores, 8GB RAM
*   **Recommended:** GPU with CUDA support for faster processing
*   **Audio:** AIOC (All-In-One-Cable) adapter, Digirig, or compatible USB audio interface
*   **Serial:** USB serial port for PTT control (typically /dev/ttyACM0, /dev/ttyACM1, or /dev/ttyUSBx)

## Testing and Development

The project includes several testing utilities:

- **Wake word testing**: Built-in debug modes in wake word detector
- **Module testing**: Each module includes `if __name__ == "__main__"` test sections
- **Command testing**: Test voice command recognition with `commands.py`

To test individual components:
```bash
python ril_aioc.py                # Test AIOC hardware interface
python ril_digirig.py             # Test Digirig hardware interface
python prompts.py                 # Test prompt management
python wake_word_detector.py      # Test wake word detection
python commands.py                # Test command identification
python periodically_identify.py   # Test periodic identification
```

## Troubleshooting

*   **Wake word not detected:** Try adjusting `AUDIO_THRESHOLD` or `AST_CONFIDENCE_THRESHOLD` in `constants.py`, or test with wake word detector debug mode
*   **Wrong wake word detected:** Choose a different wake word from the 35+ available options in `constants.py`
*   **Serial port errors:** Ensure user is in `dialout` group and device is connected
*   **Audio device not found:** Check USB connections and run `python -c "import sounddevice; print(sounddevice.query_devices())"`
*   **CUDA errors:** Install appropriate PyTorch version for your CUDA version
*   **Ollama connection errors:** Ensure Ollama service is running with `systemctl status ollama`
*   **Configuration issues:** All settings are centralized in `constants.py` - check this file first
*   **Carrier sense issues:** If PTT activation is delayed or blocked, adjust `CARRIER_SENSE_DURATION`, `AUDIO_THRESHOLD`, or disable carrier sense by setting `CARRIER_SENSE_MAX_RETRIES = 0`

## Future Enhancements

*   Automatic RIL detection
*   Online models
*   Voice APRS integration
*   Custom wake words
*   Commands to change models and online/offline modes
*   Add modularity and a command to switch from offline mode (gemma) to online mode (Gemini or ChatGPT)
*   More robust error handling and recovery
*   Dynamic loading of AI personas
*   Offline LLM support for emergency scenarios. Create an LLM with knowledge of FEMA, ARES, and other emergency services.
*   Offline multi-language and translation support

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

