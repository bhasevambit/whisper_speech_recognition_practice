# Whisper Speech-Recognition Training

This repository is Whisper Speech-Recognition Training Test Code.

## Python Version

This repository is used "**Python 3.11**".
I recommend setting up "**venv**" with python version = "3.11". 
venv setup is below commands.

`python -m venv .venv`

## whisper Library install

Please install whisper Library according to setup procedure below link.
- <https://github.com/openai/whisper#setup>


## pip Requirements

pip requirements install command is below.

`pip install -r ./requirements.txt`

## Local modules

Please git clone at `./submodules` below repositories.

- <https://github.com/bhasevambit/pyaudio_audio_signal_processing_training>

## Note

- This repository is used direnv.

  - Please install `direnv` and execute `direnv allow` commands at Repository Top directory.
    (If you use Windows, please execute `.\.venv\Scripts\activate` commnads)

- When using a Raspberry Pi or Jetson Nano, Whisper is not easy to install. Therefore, I recommend using whisper.cpp.
