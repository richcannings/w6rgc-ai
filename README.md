# W6RGC-AI: Ham Radio AI Voice Assistant

## Overview

Assuming that "AI is the new electricity". Some day we will need a backup AI system.

This experiment implements an AI backup in the form a voice assistant over ham radio. 

W6RGC-AI is a Python-based voice assistant designed for amateur radio operators. It uses advanced wake word detection, speech-to-text, a large language model (LLM), and text-to-speech to provide a conversational AI experience, integrated with ham radio operations, including PTT (Push-to-Talk) control via a serial interface (e.g., for an AIOC adapter).

The assistant features dual wake word detection methods and listens for configurable wake words before processing commands or queries, making it suitable for hands-free operation in a radio shack environment.

## Purpose

This project extends the art of amateur radio by exploring how AI can support and enhance the hobby. The goal is to demonstrate practical applications of artificial intelligence in amateur radio operations, from emergency communications to everyday radio activities.

The purposes depends on internet connectivity for full functionality. An offline solution is like an AI backup, while the online system allows for much greater functionality

### Offline

This is the AI "backup" scenario. The LLM can be trained to support amateur radio services by being another voice in solutioning by training the local LLM with ARES information, FEMA docs, County information, etc. Voice translations would be another easier feature to add. 

### Online

Connects to much more powerful LLMs. Most online LLMs can access the internet and perform many functions.

One example (and my next project) is "Voice APRS" with the features to:
* Send and receive APRS messages
* Position beaconing using AI based location (E.g. "This is W6RGC. Beacon that I am at the corner of West Cliff Drive and Swift Street in Santa Cruz")

## Features

*   **Advanced Wake Word Detection:** Dual detection methods using MIT's AST model and custom Whisper-based detection
*   **Conversational AI:** Leverages Ollama with models like Gemma3 to understand and respond to user speech
*   **High-Quality Speech Recognition:** Utilizes OpenAI Whisper for accurate transcription of spoken audio
*   **Natural Text-to-Speech:** Employs CoquiTTS for generating spoken responses with CUDA acceleration
*   **Automatic Audio Device Detection:** Automatically finds and configures AIOC (All-In-One-Cable) adapters
*   **Sample Rate Conversion:** Handles audio conversion between device rates (48kHz) and model requirements (16kHz)
*   **Customizable Persona:** AI's name, callsign, and speaking style can be configured in `prompts.py`
*   **Contextual Conversation:** Maintains conversation history for more natural interactions
*   **PTT Control:** Integrates with serial PTT for transmitting AI responses over the air
*   **Graceful Termination:** Recognizes voice commands like "break", "exit", "quit", or "shutdown" to gracefully shut down

## Wake Word Detection

The system supports two wake word detection methods:

- **AST Method:** Uses MIT's pre-trained model with 35+ available wake words (currently "seven")
- **Custom Method:** Uses Whisper for flexible custom phrases (currently "Overlord")

The default configuration uses the AST method for efficiency. To change methods or wake words, modify the detector initialization in `main.py`.

## Technologies Used

*   **Python 3**
*   **Wake Word Detection:** MIT AST (Audio Spectrogram Transformer) + Custom Whisper-based detection
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
    ```
    After installation, ensure the Ollama service is running and you have pulled a model (e.g., `ollama pull gemma3:12b`).
*   **AIOC (All-In-One-Cable) Adapter:** This project is designed to work with an AIOC adapter for PTT (Push-to-Talk) functionality. The system automatically detects AIOC devices by name.

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
    *   Review and modify parameters in `main.py` such as `THRESHOLD`, `SILENCE_DURATION`, `OLLAMA_URL`, and `MODEL`
    *   The system automatically detects AIOC hardware, but you can manually set `serial_port` if needed
    *   Customize the AI's persona and prompts in `prompts.py`
    *   Choose your wake word detection method in `main.py`

2.  **Run the main script:**
    ```bash
    python main.py
    ```

3.  **Operation:**
    *   The application will start listening for the wake word
    *   **AST Method:** Say "seven" followed by your command
    *   **Custom Method:** Say "Overlord" followed by your command
    *   Example: "Seven, what is the current UTC time?"
    *   To terminate: "Seven, break" or "Seven, exit" or use Ctrl+C

## Configuration

Key configuration files and variables:

*   **`main.py`:**
    *   `THRESHOLD`: Voice activity detection sensitivity (default: 0.02)
    *   `SILENCE_DURATION`: Duration of silence to mark end of speech (default: 2.0 seconds)
    *   `FRAME_DURATION`: Audio frame duration for VAD (default: 0.1 seconds)
    *   `OLLAMA_URL`: URL of your Ollama server (default: "http://localhost:11434/api/generate")
    *   `MODEL`: The Ollama model to use (default: "gemma3:12b")
    *   Wake word detector type: "ast" or "custom"

*   **`prompts.py`:**
    *   `BOT_NAME`: The AI's name in conversation (default: "7")
    *   `BOT_CALLSIGN`: The AI's amateur radio callsign (default: "W6RGC/AI")
    *   `BOT_SPOKEN_CALLSIGN`: How the callsign should be spoken phonetically
    *   `BOT_PHONETIC_CALLSIGN`: The verbose phonetic spelling of the callsign
    *   `PROMPT_ORIGINAL`: The base system prompt that defines the AI's persona and behavior

*   **`wake_word_detector.py`:**
    *   AST wake word selection (35+ options available)
    *   Custom wake word phrases
    *   Detection sensitivity and timing parameters

## Hardware Requirements

*   **Minimum:** CPU with 4+ cores, 8GB RAM
*   **Recommended:** GPU with CUDA support for faster processing
*   **Audio:** AIOC (All-In-One-Cable) adapter or compatible USB audio interface
*   **Serial:** USB serial port for PTT control (typically /dev/ttyACM0 or /dev/ttyACM1)

## Troubleshooting

*   **Wake word not detected:** Try adjusting `THRESHOLD` in main.py or test with `test_wake_word.py`
*   **Serial port errors:** Ensure user is in `dialout` group and device is connected
*   **Audio device not found:** Check USB connections and run `python -c "import sounddevice; print(sounddevice.query_devices())"`
*   **CUDA errors:** Install appropriate PyTorch version for your CUDA version
*   **Ollama connection errors:** Ensure Ollama service is running with `systemctl status ollama`

## Future Enhancements

*   More robust error handling and recovery
*   Configuration file (e.g., YAML or JSON) instead of hardcoded variables
*   GUI for easier configuration and status monitoring
*   Additional wake word detection engines
*   Dynamic loading of AI personas
*   Voice APRS integration
*   Offline LLM support for emergency scenarios

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
