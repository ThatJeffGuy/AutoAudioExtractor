import os
import sys
import subprocess
import logging
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler('error-log.txt')
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

# Log current system's environment variables
logging.info("Current system's environment variables:")
for key, value in os.environ.items():
    logging.info(f"{key}: {value}")

# Ensure ffmpeg is in PATH
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    logging.error("ffmpeg is not installed or not in PATH")
    error_logger.error("ffmpeg is not installed or not in PATH")
    messagebox.showerror("Error", "ffmpeg is not installed or not in PATH. Please install ffmpeg and try again.")
    sys.exit(1)
else:
    logging.info(f"ffmpeg is installed at: {ffmpeg_path}")

def create_and_activate_venv():
    if os.getenv("IN_VENV") == "1":
        logging.info("Already running in the virtual environment.")
        return

    venv_dir = tempfile.mkdtemp()
    logging.info(f"Creating virtual environment in temporary directory: {venv_dir}")
    subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])

    python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_dir, 'bin', 'python')
    logging.info(f"Python executable path: {python_executable}")

    # Uninstall any existing versions to avoid conflicts
    try:
        subprocess.check_call([python_executable, '-m', 'pip', 'uninstall', '-y', 'torch', 'torchaudio', 'torchvision'])
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to uninstall existing versions: {e}")

    # Install torch and compatible versions of torchaudio and torchvision
    try:
        subprocess.check_call([python_executable, '-m', 'pip', 'install', 'torch', 'torchaudio', 'torchvision', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install torch or related packages: {e}")
        sys.exit(1)

    # Install other dependencies
    try:
        subprocess.check_call([python_executable, '-m', 'pip', 'install', 'pyannote.audio[cuda]', 'speechbrain'])
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install other dependencies: {e}")
        sys.exit(1)

    # Check for dependency conflicts
    try:
        subprocess.check_call([python_executable, '-m', 'pip', 'check'])
    except subprocess.CalledProcessError as e:
        logging.error(f"Dependency conflicts found: {e}")
        sys.exit(1)

    logging.info(f"Re-running script with virtual environment: {python_executable}")
    env = os.environ.copy()
    env["IN_VENV"] = "1"
    result = subprocess.run([python_executable] + sys.argv, env=env)
    sys.exit(result.returncode)

def check_cuda():
    try:
        import torch
        logging.info(f"PyTorch version: {getattr(torch, '__version__', 'unknown')}")
        if not hasattr(torch, 'cuda'):
            raise ImportError("CUDA attribute not found in torch")
        if not torch.cuda.is_available():
            raise ImportError("CUDA is not available")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except ImportError as e:
        logging.error(f"CUDA check failed: {e}")
        error_logger.error(f"CUDA check failed: {e}")
        messagebox.showerror("Error", f"CUDA check failed: {e}")
        sys.exit(1)

def extract_audio(video_path, audio_path):
    logging.info(f"Extracting English audio from {video_path}")
    if os.path.exists(audio_path):
        logging.info(f"Audio file {audio_path} already exists. Removing it.")
        os.remove(audio_path)
    command = ['ffmpeg', '-i', video_path, '-map', '0:2', '-q:a', '0', audio_path]
    subprocess.run(command, check=True)
    logging.info(f"Audio extraction completed: {audio_path}")

def diarize_audio(audio_path, diarized_audio_path):
    logging.info(f"Starting speaker diarization for {audio_path}")
    from pyannote.audio import Pipeline  # Import here after ensuring the package is installed
    from speechbrain.pretrained import SpeakerRecognition  # Ensure this import is here

    if os.path.exists("diarization.txt"):
        logging.info("Diarization file diarization.txt already exists. Removing it.")
        os.remove("diarization.txt")

    if os.path.exists(diarized_audio_path):
        logging.info(f"Diarized audio file {diarized_audio_path} already exists. Removing it.")
        os.remove(diarized_audio_path)

    HUGGING_FACE_TOKEN = "your_hugging_face_token"  # Replace with your actual Hugging Face token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN, device='cuda')
    logging.info("Pipeline initialized")
    diarization = pipeline({"uri": "filename", "audio": audio_path})
    logging.info("Diarization process completed")
    
    spkrec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_dir", run_opts={"device":"cuda"})
    logging.info("SpeechBrain model loaded")
    with open("diarization.txt", "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(f"{turn.start:.1f} {turn.end:.1f} {speaker}\n")
            logging.info(f"Diarization turn: {turn.start:.1f} {turn.end:.1f} {speaker}")
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_path = f"segment_{speaker}_{int(turn.start)}.wav"
        if os.path.exists(segment_path):
            logging.info(f"Segment file {segment_path} already exists. Removing it.")
            os.remove(segment_path)
        command = ['ffmpeg', '-i', audio_path, '-ss', str(turn.start), '-to', str(turn.end), '-c', 'copy', segment_path]
        subprocess.run(command, check=True)
        segments.append(segment_path)
    with open("segments_list.txt", "w") as f:
        for segment in segments:
            f.write(f"file '{os.path.abspath(segment)}'\n")
    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'segments_list.txt', '-c', 'copy', diarized_audio_path]
    subprocess.run(command, check=True)
    logging.info("Speaker diarization completed successfully.")

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
            logging.error("No file selected. Exiting...")
            error_logger.error("No file selected. Exiting...")
            messagebox.showerror("Error", "No file selected. Exiting...")
            sys.exit()
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        diarized_audio_path = os.path.splitext(video_path)[0] + "_diarized.wav"
        try:
            extract_audio(video_path, audio_path)
            if prompt_for_diarization():
                diarize_audio(audio_path, diarized_audio_path)
                messagebox.showinfo("Success", f"Diarized audio saved as {diarized_audio_path}")
            else:
                logging.info("Diarization cancelled by user")
                messagebox.showinfo("Cancelled", "Diarization cancelled")
        except Exception as e:
            logging.error(f"Error: {e}")
            error_logger.error(f"Error: {e}")
            messagebox.showerror("Error", str(e))
            sys.exit(1)
        root.destroy()
        logging.info("Main function completed")

if __name__ == "__main__":
    main()
