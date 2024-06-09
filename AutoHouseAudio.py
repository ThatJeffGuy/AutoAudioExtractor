import os
import sys
import subprocess
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from pyAudioAnalysis import audioSegmentation as aS
import wave

# Ensure ffmpeg is in PATH
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    print("ffmpeg is not installed or not in PATH. Please install ffmpeg and try again.")
    sys.exit(1)

# Install CUDA-enabled torch, torchaudio, and torchvision globally first
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'torch==2.0.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'torchaudio==2.2.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'torchvision==0.15.2+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
except subprocess.CalledProcessError as e:
    print(f"Failed to install torch or related packages: {e}")
    sys.exit(1)

# Install pyAudioAnalysis globally
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyAudioAnalysis'])
except subprocess.CalledProcessError as e:
    print(f"Failed to install pyAudioAnalysis: {e}")
    sys.exit(1)

def create_and_activate_venv():
    if os.getenv("IN_VENV") == "1":
        return

    venv_dir = tempfile.mkdtemp()
    subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])

    python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_dir, 'bin', 'python')

    # Install other dependencies
    try:
        subprocess.check_call([python_executable, '-m', 'pip', 'install', 'speechbrain', 'soundfile', 'scipy'])
        # Reinstall torch packages to ensure CUDA compatibility within the virtual environment
        subprocess.check_call([python_executable, '-m', 'pip', 'install', '--force-reinstall', 'torch==2.0.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
        subprocess.check_call([python_executable, '-m', 'pip', 'install', '--force-reinstall', 'torchaudio==2.2.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
        subprocess.check_call([python_executable, '-m', 'pip', 'install', '--force-reinstall', 'torchvision==0.15.2+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install other dependencies: {e}")
        sys.exit(1)

    env = os.environ.copy()
    env["IN_VENV"] = "1"
    result = subprocess.run([python_executable] + sys.argv, env=env)
    sys.exit(result.returncode)

def check_cuda():
    try:
        import torch
        if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
            raise ImportError("CUDA is not available")
    except ImportError as e:
        print(f"CUDA check failed: {e}")
        sys.exit(1)

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    command = ['ffmpeg', '-i', video_path, '-map', '0:2', '-q:a', '0', audio_path]
    subprocess.run(command, check=True)

def diarize_audio(audio_path, diarized_audio_path):
    if os.path.exists("diarization.txt"):
        os.remove("diarization.txt")

    if os.path.exists(diarized_audio_path):
        os.remove(diarized_audio_path)

    flags, classes = aS.mtFileClassification(audio_path, "data/svmRBFmodel", "svm_rbf")
    segments = []
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        duration = wf.getnframes() / sample_rate
    for i, flag in enumerate(flags):
        segment_path = f"segment_{classes[flag]}_{i}.wav"
        if os.path.exists(segment_path):
            os.remove(segment_path)
        start_time = i * (duration / len(flags))
        end_time = (i + 1) * (duration / len(flags))
        command = ['ffmpeg', '-i', audio_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', segment_path]
        subprocess.run(command, check=True)
        segments.append(segment_path)
    with open("segments_list.txt", "w") as f:
        for segment in segments:
            f.write(f"file '{os.path.abspath(segment)}'\n")
    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'segments_list.txt', '-c', 'copy', diarized_audio_path]
    subprocess.run(command, check=True)
    print("Speaker diarization completed successfully.")

def prompt_for_diarization():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    result = messagebox.askyesno("Proceed with Diarization", "Do you want to proceed with automatic diarization?", parent=root)
    root.destroy()
    return result

def main():
    create_and_activate_venv()
    if os.getenv("IN_VENV") == "1":
        check_cuda()
        try:
            import speechbrain
        except ImportError as e:
            print(f"Failed to import speechbrain: {e}")
            sys.exit(1)

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        video_path = filedialog.askopenfilename(title="Select MKV File", filetypes=[("MKV files", "*.mkv")])
        if not video_path:
            print("No file selected. Exiting...")
            sys.exit()
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        diarized_audio_path = os.path.splitext(video_path)[0] + "_diarized.wav"
        try:
            extract_audio(video_path, audio_path)
            if prompt_for_diarization():
                diarize_audio(audio_path, diarized_audio_path)
                print(f"Diarized audio saved as {diarized_audio_path}")
            else:
                print("Diarization cancelled")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(0)

if __name__ == "__main__":
    main()
