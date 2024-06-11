import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog
import wave
from huggingface_hub import snapshot_download
import shutil
from pathlib import Path
from pyannote.audio import Pipeline
import torch
import torchaudio
from speechbrain.inference.interfaces import foreign_class

def ensure_model_exists(local_paths, hf_paths, use_auth_token=None):
    for local_path, hf_path in zip(local_paths, hf_paths):
        if not Path(local_path).exists():
            print(f"{local_path} not found locally. Downloading from Hugging Face: {hf_path}")
            try:
                model_dir = snapshot_download(repo_id=hf_path, use_auth_token=use_auth_token)
                if os.path.exists(local_path):
                    shutil.rmtree(local_path)
                shutil.copytree(model_dir, local_path)
                print(f"Downloaded and saved to {local_path}")
            except Exception as e:
                print(f"Error downloading {hf_path} from Hugging Face: {e}")
                sys.exit(1)
        else:
            print(f"Found expected directory: {local_path}")

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
        from speechbrain.inference import SpeakerRecognition
        print("All required libraries imported successfully.")
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        sys.exit(1)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        # Ensure the required local directories exist
        local_paths = [
            "pretrained_models/spkrec-ecapa-voxceleb",
            "pretrained_models/speakerrecognition"
        ]
        hf_paths = [
            "speechbrain/spkrec-ecapa-voxceleb",
            "pyannote/speaker-diarization@2.1"
        ]
        ensure_model_exists(local_paths, hf_paths, use_auth_token="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")

        # Check for custom.py file
        for local_path in local_paths:
            custom_py_path = os.path.join(local_path, 'custom.py')
            if not Path(custom_py_path).exists():
                print(f"Expected custom.py not found in: {local_path}")
                sys.exit(1)
            else:
                print(f"Found custom.py in: {local_path}")

        # Initialize the SpeechBrain model using foreign_class
        sb_local_path = "pretrained_models/spkrec-ecapa-voxceleb"
        classifier = foreign_class(
            source="speechbrain/spkrec-ecapa-voxceleb",
            pymodule_file=os.path.join(sb_local_path, "custom.py"),
            classname="CustomEncoderWav2vec2Classifier"
        )

        signal, fs = torchaudio.load(audio_path, backend='sox_io')
        embeddings = classifier.encode_batch(signal)
        print("SpeechBrain model initialized and embeddings extracted successfully.")

        # Initialize the pyannote pipeline
        pa_local_path = "pretrained_models/speakerrecognition"
        pipeline = Pipeline.from_pretrained(pa_local_path, use_auth_token="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")
        pipeline.to(device)
    except Exception as e:
        print(f"Error loading the models: {e}")
        sys.exit(1)

    diarization = pipeline(audio_path)

    # Save diarization result to file
    with open(os.path.join(segments_folder, "diarization.rttm"), "w") as f:
        diarization.write_rttm(f)

    # Extract segments and save them as individual files
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

def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title="Select File", filetypes=[("All files", "*.*")])
    if not file_path:
        print("No file selected. Exiting...")
        sys.exit()

    file_dir = os.path.dirname(file_path)
    segments_folder = os.path.join(file_dir, "segments")
    os.makedirs(segments_folder, exist_ok=True)

    file_ext = os.path.splitext(file_path)[1].lower()
    audio_path = os.path.splitext(file_path)[0] + ".wav"
    diarized_audio_path = os.path.join(segments_folder, os.path.splitext(os.path.basename(file_path))[0] + "_diarized.wav")

    if file_ext in ['.mkv', '.mp4', '.avi', '.mov']:
        print("Extracting audio from video file...")
        extract_audio(file_path, audio_path)
    elif file_ext in ['.wav']:
        print("Audio file is already in WAV format.")
        audio_path = file_path
    elif file_ext in ['.mp3', '.flac', '.ogg']:
        print("Converting audio file to WAV format...")
        convert_audio(file_path, audio_path)
    else:
        print(f"Unsupported file format: {file_ext}. Exiting...")
        sys.exit()

    print("Starting diarization...")
    diarize_audio(audio_path, diarized_audio_path, segments_folder)
    print(f"Diarized audio saved as {diarized_audio_path}")

if __name__ == "__main__":
    main()
