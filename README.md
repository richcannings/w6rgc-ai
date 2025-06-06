# W6RGC/AI: On and off grid ham radio AI voice assistant

## Overview

AI is changing everything. Operator: Are you ready?

W6RGC/AI is an experiment applying AI to ham radio in the form of an AI voice assistant, and runs an assortment of online and fully offline models.

Your ham radio is the user interface. When on the air, activate W6RGC/AI by saying the wake word (like "Seven") at the start of every transmission, and then tell "Seven" what you need or ask what it does.

Prompt based features include:
*   Perform QSOs (exchange call sign, name, location, and signal report with confirmation)
*   Run simple nets (handle corrections, maintain list, count)
*   [FEMA ICS-213](https://training.fema.gov/icsresource/icsforms.aspx) message passing and recording

Function based features include:
*   Voice-based [APRS](https://www.aprs.org/): Natural language APRS message sending/receiving
*   Basic voice commands using regular expressions, like "status", "exit", "identify".

Offline mode allows to run this as an AI backup system, for the day when AI becomes as a necessity as electricity and communication.

Works with the [Digirig](https://digirig.net/) and [AIOC](https://github.com/skuep/AIOC) ham radio adapters.

## Design

W6RGC/AI explores and leverages the following AI models in concert:

1.  **AI based wake word spotting** 
    - Uses [MIT/AST](https://huggingface.co/MIT/ast-finetuned-speech-commands-v2)
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
    - Copying FEMA ICS-213 forms
5.  **AI based tooling and function calling**
    - Uses [Gemini Function Calling](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting)
    - Example: "Voice APRS" sends and receive APRS messages using Natural Language Understanding.
    - A regular expression based system is available too 
6.  **AI based text-to-speech** using [CoquiTTS](https://github.com/coqui-ai/TTS) to give the bot a voice

The code is designed for modularity, making it easy to swap and compare AI models, prompts, and function calling. The system is organized into several key modules:

- **`main.py`**: Main application entry point and orchestration
- **`constants.py`**: Centralized configuration management for all settings
- **`regex_command_tooling.py`**: Command identification and parsing for voice commands (status, reset, identify, terminate)
- **`ril_aioc.py`**: Radio Interface Layer for AIOC hardware management
- **`ril_digirig.py`**: Radio Interface Layer for Digirig hardware management
- **`prompts.py`**: AI persona and conversation management (class-based)
- **`wake_word_detector.py`**: AST-based wake word detection system
- **`periodically_identify.py`**: Handles periodic station identification

![Architecture Diagram](img/architecture.jpg)

## Features

W6RGC/AI offers a range of features leveraging various AI technologies:

**Prompt-Based Features:**
*   **Perform QSOs:** Exchange call sign, name, location, and signal reports with confirmation.
*   **Run Simple Nets:** Handle check-ins, corrections, maintain a participant list, and provide counts.
*   **FEMA ICS-213 Messaging:** Assist with passing and recording messages in the FEMA ICS-213 format.

**Function-Based Features:**
*   **VoiceAPRS:** Send and receive APRS messages using natural language voice commands (requires internet).
*   **Basic Voice Commands:** Regex-based commands for "status," "exit," "reset," and "identify."

**Core Technical Features:**
*   **Efficient Wake Word Detection:** Uses MIT's AST (Audio Spectrogram Transformer) model.
*   **High-Quality Speech Recognition:** Utilizes OpenAI Whisper.
*   **Conversational AI:** Leverages modular LLMs (Online: Gemini, Offline: Ollama).
*   **Natural Text-to-Speech:** Employs CoquiTTS with CUDA acceleration.
*   **Automatic Audio Device Detection:** For AIOC and Digirig adapters.
*   **Modular Architecture:** Facilitates easy swapping of AI models, prompts, and hardware interfaces.
*   **Centralized Configuration:** All settings managed via `constants.py`.
*   **PTT Control & Carrier Sense:** For controlled radio transmission.
*   **Periodic Identification:** Automatic station ID.
*   **Graceful Termination:** Via voice commands.

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

## Operation

Your radio is the main interface for input and output. The application will start and listen for the configured wake word (default: "Seven"). To interact, say the wake word at the start of your transmission once or twice, followed by your command or query speaking as how you would speak with another operator.

Example: "Seven, what is the current UTC time?" or "Seven, send an APRS message."

To terminate the assistant: "Seven, break" or "Seven, exit" or use Ctrl+C in the terminal.

## Wake Word Detection and Changing Wake Word

The system uses MIT's AST (Audio Spectrogram Transformer) model for efficient wake word detection:

- **35+ Available Wake Words:** backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero
- **Current Default:** "seven" (optimized for ham radio use)
- **High Performance:** Very fast, low CPU usage, high accuracy
- **CUDA Support:** GPU acceleration when available

To change the wake word, modify `DEFAULT_WAKE_WORD` in `constants.py` to any of the supported words.

## Configuration

The application uses a centralized configuration system through `constants.py`. Key configuration sections include:

### AI/LLM Configuration
```python
# Internet Connectivity and LLM Selection
HAS_INTERNET = True  # Set to True for Gemini (online), False for Ollama (offline)

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OFFLINE_MODEL = "gemma3:12b"  # Alternative: "gemma3:4b"

# Gemini Configuration
GEMINI_API_KEY_FILE = "gemini_api_key.txt"  # Store your Gemini API key in this file
DEFAULT_ONLINE_MODEL = "models/gemini-2.5-flash-preview-05-20" # Example: "gemini-1.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
REQUEST_TIMEOUT = 30  # seconds
```

### Wake Word Detection
```python
# AST Wake Word Detection
DEFAULT_WAKE_WORD = BOT_NAME # syncing bot name with wake word
AST_CONFIDENCE_THRESHOLD = 0.7
AST_CHUNK_LENGTH_S = 1.0
AST_MODEL_NAME = "MIT/ast-finetuned-speech-commands-v2"

# Wake Word Detection Method
# WAKE_WORD_METHOD_AST = "ast" # This is an internal constant, usually not changed by user
DEFAULT_WAKE_WORD_METHOD = "ast" # Currently "ast" is the primary method

# Command Detection Configuration
MAX_COMMAND_WORDS = 10  # maximum words to check for commands (prevents accidental triggers)
```

### Hardware Configuration
```python
# Radio Interface Layer (RIL) Configuration
# RIL_TYPE_AIOC = "aioc" # Internal constant
# RIL_TYPE_DIGIRIG = "digirig" # Internal constant
DEFAULT_RIL_TYPE = "digirig"  # Options: "aioc" or "digirig". Change this to switch.

# Serial Port Configuration
DEFAULT_AIOC_SERIAL_PORT = "/dev/ttyACM0"  # Serial port for AIOC PTT control
DEFAULT_DIGIRIG_SERIAL_PORT = "/dev/ttyUSB2" # Serial port for Digirig PTT control
SERIAL_TIMEOUT = 1  # seconds
```

### Bot Identity (also configurable in prompts.py)
```python
OPERATOR_NAME = "Operator"
BOT_NAME = "seven" # Here are the following wake word / bot name options: seven (7), marvin , shiela, zero (0), happy, forward.
BOT_CALLSIGN = "W6RGC/AI"      # Amateur radio callsign
BOT_SPOKEN_CALLSIGN = "W 6 R G C stroke I A"
BOT_PHONETIC_CALLSIGN = "Whiskey 6 Romeo Golf Charlie Stroke Alpha India"
```

### TTS Configuration
```python
# Script File Path for logging conversation
WRITE_SCRIPT_TO_FILE = True
# SCRIPT_FILE_PATH = f"chatbot-script-{_date_str}-{_time_str}.log" # Dynamically generated if WRITE_SCRIPT_TO_FILE is True

# TTS Model Options
TTS_MODEL_FAST_PITCH = "tts_models/en/ljspeech/fast_pitch"     # Fastest TTS model
TTS_MODEL_SPEEDY_SPEECH = "tts_models/en/ljspeech/speedy_speech" # Alternative fast model
TTS_MODEL_TACOTRON2 = "tts_models/en/ljspeech/tacotron2-DDC"   # Fallback model

# TTS Audio Settings
TTS_INFERENCE_NOISE_SCALE = 0.667
TTS_INFERENCE_NOISE_SCALE_DP = 1.0
TTS_INFERENCE_SIGMA = 1.0

# TTS File Configuration
TTS_OUTPUT_FILE = 'ollama_tts.wav' # Temporary file for TTS output
```

### Carrier Sense Configuration
```python
CARRIER_SENSE_DURATION = 0.5      # seconds to monitor for carrier before PTT
CARRIER_SENSE_MAX_RETRIES = 3     # maximum attempts to find clear frequency
CARRIER_SENSE_RETRY_DELAY = 3.0   # seconds to wait between carrier sense attempts
```

### Periodic Identification Configuration
```python
PERIODIC_ID_INTERVAL_MINUTES = 10  # minutes between automatic identification announcements
```

## Hardware Requirements

*   **Minimum:** CPU with 4+ cores, 32GB ram, 10GB disk (not including ollama models)
*   **GPU with CUDA support** RTX3060 with 12GB RAM. More GPUs, the better.
*   **Radio interface** AIOC (All-In-One-Cable) adapter, Digirig, or compatible USB audio interface

## Testing and Development

The project includes several testing utilities:

- **Wake word testing**: Built-in debug modes in wake word detector
- **Module testing**: Each module includes `if __name__ == "__main__"` test sections
- **Command testing**: Test voice command recognition with `regex_command_tooling.py`

To test individual components:
```bash
python ril_aioc.py                # Test AIOC hardware interface
python ril_digirig.py             # Test Digirig hardware interface
python prompt_gemini_generated.py # Test prompt management 
python prompt_original.py         # Test prompt management 
python wake_word_detector.py      # Test wake word detection
python regex_command_tooling.py   # Test command identification
python periodically_identify.py   # Test periodic identification
python list_gemini_models.py      # Utility to list available Gemini models or verify API access
```

## Troubleshooting

*   **Wake word not detected:** Try adjusting `AUDIO_THRESHOLD` or `AST_CONFIDENCE_THRESHOLD` in `constants.py`, or test with wake word detector debug mode
*   **Wrong wake word detected:** Choose a different wake word from the 35+ available options in `constants.py`
*   **Serial port errors:** Ensure user is in `dialout` group and device is connected
*   **Audio device not found:** Check USB connections and run `python -c "import sounddevice; print(sounddevice.query_devices())"`
*   **CUDA errors:** Install appropriate PyTorch version for your CUDA version
*   **Ollama connection errors:** Ensure Ollama service is running with `systemctl status ollama`
*   **Configuration issues:** All settings are centralized in `constants.py`