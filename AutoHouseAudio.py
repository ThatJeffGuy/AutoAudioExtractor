import os
import sys
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog
import wave
import pkg_resources

def install_package(package_name, version=None, extra_index_url=None):
    try:
        if version:
            if extra_index_url:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}=={version}", "-f", extra_index_url])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}=={version}"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")
        sys.exit(1)

def is_package_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def ensure_dependencies():
    if not is_package_installed('torch'):
        install_package('torch', '2.0.1+cu118', 'https://download.pytorch.org/whl/torch_stable.html')
    if not is_package_installed('torchaudio'):
        install_package('torchaudio', '2.0.1+cu118', 'https://download.pytorch.org/whl/torch_stable.html')
    install_package('pyannote.audio')
    install_package('speechbrain')
    install_package('soundfile')

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
    from pyannote.audio import Pipeline
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    pipeline.to(device)

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
    ensure_dependencies()

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
