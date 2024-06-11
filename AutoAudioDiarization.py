import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog
import wave
from pathlib import Path
import shutil
from pyannote.audio import Pipeline
import torch
import torchaudio
from speechbrain.inference.interfaces import foreign_class

def ensure_model_exists(local_paths):
    for local_path in local_paths:
        if not Path(local_path).exists():
            print(f"Expected directory not found: {local_path}")
            print("Please ensure the following directories are available locally:")
            for path in local_paths:
                print(f"  - {path}")
            sys.exit(1)
        else:
            print(f"Found expected directory: {local_path}")
        
        custom_py_path = os.path.join(local_path, 'custom.py')
        if not Path(custom_py_path).exists():
            print(f"Expected custom.py not found in: {local_path}")
            sys.exit(1)
        else:
            print(f"Found custom.py in: {local_path}")
        
        hyperparams_path = os.path.join(local_path, 'hyperparams.yaml')
        if not Path(hyperparams_path).exists():
            print(f"Expected hyperparams.yaml not found in: {local_path}")
            sys.exit(1)
        else:
            print(f"Found hyperparams.yaml in: {local_path}")

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path]
    subprocess.run(command, check=True)

def convert_audio(audio_path, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    command = ['ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_path]
    subprocess.run(command, check=True)

def diarize_audio(audio_path, diarized_audio_path, segments_folder):
    try:
        print("Importing libraries...")
        from pyannote.audio import Pipeline
        import torch
        import torchaudio
        from speechbrain.inference.interfaces import foreign_class  # Use this import
        print("All required libraries imported successfully.")
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        local_paths = [
            "pretrained_models/spkrec-ecapa-voxceleb",
            "pretrained_models/speakerrecognition",
            "pretrained_models/customencoderwav2vec2classifier"
        ]
        ensure_model_exists(local_paths)

        for local_path in local_paths:
            custom_py_path = os.path.join(local_path, 'custom.py')
            if not Path(custom_py_path).exists():
                print(f"Expected custom.py not found in: {local_path}")
                sys.exit(1)
            else:
                print(f"Found custom.py in: {local_path}")
            
            hyperparams_path = os.path.join(local_path, 'hyperparams.yaml')
            if not Path(hyperparams_path).exists():
                print(f"Expected hyperparams.yaml not found in: {local_path}")
                sys.exit(1)
            else:
                print(f"Found hyperparams.yaml in: {local_path}")

        sb_local_path = "pretrained_models/spkrec-ecapa-voxceleb"
        classifier_sb = foreign_class(
            source=sb_local_path,
            pymodule_file=os.path.join(sb_local_path, "custom.py"),
            classname="CustomEncoderWav2Vec2Classifier",
            savedir=sb_local_path
        )

        pa_local_path = "pretrained_models/speakerrecognition"
        classifier_pa = foreign_class(
            source=pa_local_path,
            pymodule_file=os.path.join(pa_local_path, "custom.py"),
            classname="CustomSpeakerRecognition",
            savedir=pa_local_path
        )

        signal, fs = torchaudio.load(audio_path, backend='sox_io')
        embeddings_sb = classifier_sb.encode_batch(signal)
        embeddings_pa = classifier_pa.encode_batch(signal)
        print("SpeechBrain and pyannote models initialized and embeddings extracted successfully.")

    except Exception as e:
        print(f"Error loading the models: {e}")
        sys.exit(1)

    diarization = classifier_pa.classify_file(audio_path)

    with open(os.path.join(segments_folder, "diarization.rttm"), "w") as f:
        diarization.write_rttm(f)

    segments = []
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_path = os.path.join(segments_folder, f"segment_{speaker}_{turn.start:.3f}_{turn.end:.3f}.wav")
            start_time = turn.start
            end_time = turn.end
            command = ['ffmpeg', '-i', audio_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', segment_path]
            subprocess.run(command, check=True)
            segments.append(segment_path)

    with open(os.path.join(segments_folder, "segments_list.txt"), "w") as f:
        for segment in segments:
            f.write(f"file '{os.path.abspath(segment)}'\n")

    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', os.path.join(segments_folder, "segments_list.txt"), '-c', 'copy', diarized_audio_path]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    main()
