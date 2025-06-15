# Piper TTS Voice Models

This directory stores the Piper TTS voice models used by the application. To use Piper TTS, you must download at least one voice model.

The application is configured to fall back to the Coqui TTS engine if the Piper model files are not found.

## Required Files

For each voice model, you need two files:
- The model data file: `*.onnx`
- The model configuration file: `*.onnx.json`

Both files must be present in this directory for the chosen voice to work correctly.

## Default Model: `en_US-lessac-medium`

The `constants.py` file is configured to use the `en_US-lessac-medium` model by default.

To download this model, run the following commands from inside the `piper-tts-models` directory:
```bash
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

## Other Recommended Models

You can find many other voices on the [Piper Voices Hugging Face repository](https://huggingface.co/rhasspy/piper-voices/tree/main).

Here are a few other popular options. Remember to run these commands from this directory.

### Southern English Female (Low Quality)
- **Model:** `en_GB-southern_english_female-low`
- **Commands:**
  ```bash
  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/southern_english_female/low/en_GB-southern_english_female-low.onnx
  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/southern_english_female/low/en_GB-southern_english_female-low.onnx.json
  ```

### US English LJSpeech (Medium Quality - Similar to Coqui TTS)
- **Model:** `en_US-ljspeech-medium`
- **Commands:**
  ```bash
  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx
  wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json
  ```

## Using a Different Model

If you download a model other than the default (`en_US-lessac-medium`), you **must** update the `TTS_PIPER_MODEL_PATH` variable in `constants.py` to point to the new `.onnx` file path.

For example, if you choose the `en_GB-southern_english_female-low` model, change the line in `constants.py` to:
```python
# constants.py

TTS_PIPER_MODEL_PATH = "piper-tts-models/en_GB-southern_english_female-low.onnx"
```
