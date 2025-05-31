# W6RGC/AI: On and off grid ham radio AI voice assistant

## Overview

AI is changing everything. Are you ready, Operator?

As keyboards lose favor to natural language input (your regular voice), I believe ham radio will become even more relevant because ham radios are great at transmitting and receving natural language. And, like the ham radio operator, AI has the potential to be a great pairing of intelligence and communication. That is how W6RGC/AI was born.

W6RGC/AI is an experiment applying AI to ham radio in the form of an AI voice assistant. 

W6RGC/AI explore and leverages the following AI models in concert:

1.  **AI based wake word spotting** 
    - Uses [MIT/AST](https://huggingface.co/MIT/ast-finetuned-speech-commands-v2):
    -   Listens for "Seven" by default (like "Are you Ready?" in [Western Union 92 codes](https://en.wikipedia.org/wiki/Wire_signal))
    -   You can choose from over 35 not-so-good options
2.  **AI based speech-to-text**
    - Uses [OpenAI Whisper](https://github.com/openai/whisper) speech recognition
3.  **Modular LLMs for the brains of the operation** 
    - The current default uses online [Gemini](https://gemini.google.com/) "[gemini-2.5-flash-preview-05-20](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash)" and requires a developer key
    - A one line change (`HAS_INTERNET`) switches to offline [Ollama](https://ollama.com/) model, like "[gemma3:12b](https://ollama.com/bsahane/gemma3:12b)"
    - Plug in your own models and prompts!
4.  **Ham radio prompts**. Example prompts for:
    - Performing QSOs
    - Running nets
    - recording FEMA ICS-213
5.  **AI based tooling and function calling**
    - Uses[Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting)
    - Optional regular expression based system too 
    - Example: "Voice APRS" sends and receive APRS messages using Natural Language Understanding.
6.  **AI based text-to-speech** using [CoquiTTS](https://github.com/coqui-ai/TTS) to give the bot a voice

The app communicates with ham radios with a:
- [AIOC adapter](https://github.com/skuep/AIOC)
- [Digirig](https://digirig.net/).

For your tinkering pleasure, code is designed for modularity, making it east to swap and compare AI models, prompts, and function calling.

Your ham radio is the user interface. When on the air, activate W6RGC/AI by saying the wake word (like "Seven") at the start of every transmission, and then tell "Seven" what you need.

## Purpose

This project is all about mixing the ever-relevant hobby of amateur radio with the new world of AI. The main idea is to see how AI can make ham radio even better, whether that's for critical communications during an emergency, or just for fun, everyday radio chats.

Think of it this way:
*   **When the internet's down (Offline):** This is your AI backup. Imagine an LLM trained with all sorts of useful info for ham radio operators – ARES manuals, FEMA documents, local emergency plans, and so on. It could be an extra "voice" helping you figure things out. Plus, things like voice translation could be added pretty easily.
*   **When you're online:** You can connect to even more powerful LLMs. Most online AIs can browse the internet and do all sorts of cool stuff through Natural Language Understanding.

**New: VoiceAPRS Functionality!**
The latest version now includes VoiceAPRS, allowing operators to send and receive APRS messages using natural language voice commands. This feature leverages internet connectivity to interact with APRS services (via findu.com), providing a seamless voice-controlled APRS experience. You can ask the assistant to read your messages or send a message to another callsign, all through voice.

Next up is voice APRS beaconing. Imagine saying "This is W6RGC. I'm at the corner of West Cliff Drive and Swift Street in Santa Cruz. Beacon my position on APRS." Then the chatbot uses Google Places to identify your GPS coordinates and radius, then beacons. 

## Features

*   **VoiceAPRS**: Send and receive APRS messages using natural language voice commands, leveraging NLP and internet connectivity.
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

## Setup and Installation

### Prerequisites

*   **CUDA Hardware** For GPU acceleration. This project runs local LLMs and requires dedicated GPU hardware 
    Minimally, a [Nvidia GeoForce RTX 3060](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/) is required to run all these models.
*   **Audio hardware with PTT**. Digirig and AIOC (All-In-One-Cable) are supported
*   **Operating System:** This project only been tested on Linux
*   **Required Linux Packages:** Install the following packages using apt:
    ```bash
    sudo apt update
    sudo apt install python3-full portaudio19-dev ffmpeg
    ```
*   **User Permissions:** Add your user to the `dialout` group for serial port access:
    ```bash
    sudo usermod -a -G dialout $USER
    ```
    Log out and back in for changes to take effect
*   **Ollama:** Install Ollama by running the following command in your terminal:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```
    After installation, ensure the Ollama service is running and you have pulled a model (e.g., `ollama pull gemma3:12b`).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/richcannings/w6rgc-ai
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
DEFAULT_OFFLINE_MODEL = "gemma3:12b"    # Ollama model to use
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

*   **Minimum:** CPU with 4+ cores, 32GB RAM, 
*   **GPU with CUDA support** RTX3060 with 12GB RAM. More GPUs, the better.
*   **Audio:** AIOC (All-In-One-Cable) adapter, Digirig, or compatible USB audio interface

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

