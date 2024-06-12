import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog
import wave
from pathlib import Path
import torch
import torchaudio
from pyannote.audio import Pipeline
from speechbrain.inference import EncoderClassifier

# Set the environment variables for model paths
os.environ['PYANNOTE_SPEAKER_DIARIZATION'] = "pyannote/speaker-diarization"
os.environ['SPKREC_ECAPA_VOXCELEB'] = "D:/python_programs/autoautoextractor/pretrained_models/spkrec-ecapa-voxceleb"

class CustomEncoderWav2vec2Classifier:
    def __init__(self, source=None, hparams_file="D:/python_programs/autoautoextractor/pretrained_models/hyperparams.yaml", savedir=None, run_opts=None, overrides=None):
        self.classifier = EncoderClassifier.from_hparams(
            source=source or "D:/python_programs/autoautoextractor/pretrained_models/spkrec-ecapa-voxceleb",
            hparams_file=hparams_file,
            savedir=savedir,
            run_opts=run_opts,
            overrides=overrides,
            local_files_only=True  # Ensure loading from local files only
        )
    
    def encode_batch(self, wavs):
        embeddings = self.classifier.encode_batch(wavs)
        return embeddings
    
    def classify_batch(self, wavs):
        predictions = self.classifier.classify_batch(wavs)
        return predictions

    def classify_file(self, file_path):
        predictions = self.classifier.classify_file(file_path)
        return predictions

def ensure_model_exists(local_paths):
    """
    Ensure that the required model directories and files exist locally.
    """
    for local_path in local_paths:
        local_path = Path(local_path).resolve()
        if not local_path.exists():
            print(f"Expected directory not found: {local_path}")
            print("Please ensure the following directories are available locally:")
            for path in local_paths:
                print(f"  - {path}")
            sys.exit(1)
        else:
            print(f"Found expected directory: {local_path}")

        custom_py_path = local_path / 'custom.py'
        hyperparams_path = local_path / 'hyperparams.yaml'

        if not custom_py_path.exists():
            print(f"Expected custom.py not found in: {custom_py_path}")
            sys.exit(1)
        else:
            print(f"Found custom.py in: {custom_py_path}")

        if not hyperparams_path.exists():
            print(f"Expected hyperparams.yaml not found in: {hyperparams_path}")
            sys.exit(1)
        else:
            print(f"Found hyperparams.yaml in: {hyperparams_path}")

def extract_audio(video_path, audio_path):
    """
    Extract audio from a video file using ffmpeg.
    """
    if audio_path.exists():
        audio_path.unlink()
    command = ['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_path)]
    subprocess.run(command, check=True)

def convert_audio(audio_path, output_path):
    """
    Convert an audio file to WAV format using ffmpeg.
    """
    if output_path.exists():
        output_path.unlink()
    command = ['ffmpeg', '-i', str(audio_path), '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(output_path)]
    subprocess.run(command, check=True)

def initialize_models():
    """
    Initialize the SpeechBrain and pyannote models.
    """
    try:
        # Initialize the SpeechBrain model directly
        classifier_sb = CustomEncoderWav2vec2Classifier(
            hparams_file="D:/python_programs/autoautoextractor/pretrained_models/hyperparams.yaml",
            savedir="D:/python_programs/autoautoextractor/pretrained_models"
        )

        # Initialize the pyannote pipeline
        pipeline = Pipeline.from_pretrained(os.environ['PYANNOTE_SPEAKER_DIARIZATION'])
    except Exception as e:
        print(f"Error initializing models: {e}")
        sys.exit(1)

    return classifier_sb, pipeline

def diarize_audio(audio_path, diarized_audio_path, segments_folder, classifier_sb, pipeline):
    """
    Perform audio diarization and save the diarized segments.
    """
    signal, fs = torchaudio.load(str(audio_path), backend='sox_io')
    embeddings_sb = classifier_sb.encode_batch(signal)
    print("SpeechBrain and pyannote models initialized and embeddings extracted successfully.")

    diarization = pipeline({'uri': audio_path, 'audio': audio_path})

    rttm_path = segments_folder / "diarization.rttm"
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)

    segments = []
    with wave.open(str(audio_path), 'rb') as wf:
        sample_rate = wf.getframerate()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_filename = f"segment_{speaker}_{turn.start:.3f}_{turn.end:.3f}.wav"
            segment_path = segments_folder / segment_filename
            start_time = turn.start
            end_time = turn.end
            command = ['ffmpeg', '-i', str(audio_path), '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', str(segment_path)]
            subprocess.run(command, check=True)
            segments.append(segment_path)

    segments_list_path = segments_folder / "segments_list.txt"
    with open(segments_list_path, "w") as f:
        for segment in segments:
            f.write(f"file '{segment.resolve()}'\n")

    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', str(segments_list_path), '-c', 'copy', str(diarized_audio_path)]
    subprocess.run(command, check=True)

def main():
    """
    Main function to run the audio diarization pipeline.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title="Select File", filetypes=[("All files", "*.*")])
    if not file_path:
        print("No file selected. Exiting...")
        sys.exit()

    file_path = Path(file_path).resolve()
    file_dir = file_path.parent
    segments_folder = file_dir / "segments"
    segments_folder.mkdir(exist_ok=True)

    file_ext = file_path.suffix.lower()
    audio_path = file_path.with_suffix(".wav")
    diarized_audio_path = segments_folder / f"{file_path.stem}_diarized.wav"

    if file_ext in ['.mkv', '.mp4', '.avi', '.mov']:
        print("Extracting audio from video file...")
        extract_audio(file_path, audio_path)
    elif file_ext == '.wav':
        print("Audio file is already in WAV format.")
    elif file_ext in ['.mp3', '.flac', '.ogg']:
        print("Converting audio file to WAV format...")
        convert_audio(file_path, audio_path)
    else:
        print(f"Unsupported file format: {file_ext}. Exiting...")
        sys.exit()

    try:
        classifier_sb, pipeline = initialize_models()
    except Exception as e:
        print(f"Error initializing models: {e}")
        sys.exit(1)

    print("Starting diarization...")
    diarize_audio(audio_path, diarized_audio_path, segments_folder, classifier_sb, pipeline)
    print(f"Diarized audio saved as {diarized_audio_path}")

if __name__ == "__main__":
    main()
