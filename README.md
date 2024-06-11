# AutoHouseAudio

## Overview

AutoHouseAudio is a Python-based tool designed for extracting, converting, and performing speaker diarization on audio from various video and audio files. This script utilizes `ffmpeg`, `pyannote.audio`, and `speechbrain` to provide a seamless process for handling audio files and extracting individual speaker segments.

## Features

- Extracts audio from video files in various formats (e.g., mkv, mp4, avi, mov).
- Converts audio files to WAV format if they are in unsupported formats (e.g., mp3, flac, ogg).
- Performs speaker diarization using `pyannote.audio` and `speechbrain`.
- Saves diarized segments as individual audio files and provides a combined diarized audio file.

## Installation

To use AutoHouseAudio, you need to have Python installed along with several dependencies. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/AutoHouseAudio.git
    cd AutoHouseAudio
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` should include:
    ```
    ffmpeg
    pyannote.audio
    speechbrain
    torch
    torchvision
    torchaudio
    tkinter
    ```

3. Ensure you have `ffmpeg` installed on your system. You can download it from [FFmpeg](https://ffmpeg.org/download.html).

## Usage

Run the script using Python:

```bash
python AutoHouseAudio.py
