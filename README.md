# AutoAudio Diarization

## Overview

AutoAudio Diarization is a Python-based tool designed for extracting, converting, and performing speaker diarization on audio from various video and audio files. This script utilizes `ffmpeg`, `pyannote.audio`, and `speechbrain` to provide a seamless process for handling audio files and extracting individual speaker segments.

## CPU and CUDA environments

This is the CUDA branch.

```bash
git checkout cuda
```

I am splitting this project up into two branches, one that will run entirely on CPU, and one that will have CUDA support.

CUDA is signficantly more complicated and reliant on so many libraries that cross reference specific versions that it becomes a mess to navigate and process correctly or efficiently.

CPU branch, while slower, is still very fast, with average of about 90 seconds for 1 hour of processed audio (meaning, cleaned up manually), on an i7 11700k.

## Features

- Extracts audio from video files in various formats (e.g., mkv, mp4, avi, mov).
- Converts audio files to WAV format if they are in unsupported formats (e.g., mp3, flac, ogg).
- Performs speaker diarization using `pyannote.audio` and `speechbrain`.
- Saves diarized segments as individual audio files and provides a combined diarized audio file.

## Installation

To use AutoAudio Diarization, you need to have Python installed along with several dependencies. Follow the steps below to set up your environment:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/AutoAudioDiarization.git
    cd AutoAudioDiarization
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
python AutoAudioDiarization.py
```
