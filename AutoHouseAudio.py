import os
import sys
import subprocess
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

# Ensure ffmpeg is in PATH
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    print("ffmpeg is not installed or not in PATH. Please install ffmpeg and try again.")
    sys.exit(1)

# Install CUDA-enabled torch, torchaudio, and torchvision globally first
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'torch==2.0.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'torchaudio==2.0.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', 'torchvision==0.15.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
except subprocess.CalledProcessError as e:
    print(f"Failed to install torch or related packages: {e}")
    sys.exit(1)

def create_and_activate_venv():
    if os.getenv("IN_VENV") == "1":
        return

    venv_dir = tempfile.mkdtemp()
    subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])

    python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_dir, 'bin', 'python')

    # Install other dependencies
    try:
        subprocess.check_call([python_executable, '-m', 'pip', 'install', 'pyannote.audio[cuda]', 'speechbrain'])
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
    from pyannote.audio import Pipeline
    from speechbrain.pretrained import SpeakerRecognition

    if os.path.exists("diarization.txt"):
        os.remove("diarization.txt")

    if os.path.exists(diarized_audio_path):
        os.remove(diarized_audio_path)

    HUGGING_FACE_TOKEN = "your_hugging_face_token"
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN, device='cuda')
    diarization = pipeline({"uri": "filename", "audio": audio_path})
    
    spkrec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_dir", run_opts={"device":"cuda"})
    with open("diarization.txt", "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(f"{turn.start:.1f} {turn.end:.1f} {speaker}\n")
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_path = f"segment_{speaker}_{int(turn.start)}.wav"
        if os.path.exists(segment_path):
            os.remove(segment_path)
        command = ['ffmpeg', '-i', audio_path, '-ss', str(turn.start), '-to', str(turn.end), '-c', 'copy', segment_path]
        subprocess.run(command, check=True)
        segments.append(segment_path)
    with open("segments_list.txt", "w") as f:
        for segment in segments:
            f.write(f"file '{os.path.abspath(segment)}'\n")
    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'segments_list.txt', '-c', 'copy', diarized_audio_path]
    subprocess.run(command, check=True)

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

if __name__ == "__main__":
    main()
