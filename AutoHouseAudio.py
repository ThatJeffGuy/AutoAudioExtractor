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
    venv_dir = tempfile.mkdtemp()
    logging.info(f"Creating virtual environment in temporary directory: {venv_dir}")
    subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])

    python_executable = os.path.join(venv_dir, 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(venv_dir, 'bin', 'python')
    logging.info(f"Python executable path: {python_executable}")

    if sys.executable != python_executable:
        logging.info(f"Re-running script with virtual environment: {python_executable}")
        result = subprocess.run([python_executable] + sys.argv, env={**os.environ, "IN_VENV": "1"})
        sys.exit(result.returncode)

def is_package_installed(package_name):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'show', package_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def uninstall_package(package_name):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])
        logging.info(f"Uninstalled {package_name}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to uninstall {package_name}: {e}")
        error_logger.error(f"Failed to uninstall {package_name}: {e}")
        messagebox.showerror("Error", f"Failed to uninstall {package_name}. Please check the logs for more details.")
        sys.exit(1)

def install_packages():
    packages = [
        ('torch', '2.0.1+cu117'),
        ('torchaudio', '2.0.1+cu117'),
        ('torchvision', '0.15.2+cu117'),
        'pydub',
        'pyannote.audio[cuda]',
        'speechbrain'
    ]
    for package in packages:
        if isinstance(package, tuple):
            package_name, package_version = package
            if not is_package_installed(package_name) or 'cpu' in subprocess.check_output([sys.executable, '-m', 'pip', 'show', package_name]).decode().lower():
                if is_package_installed(package_name):
                    uninstall_package(package_name)
                logging.info(f"Installing {package_name}=={package_version}")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', f'{package_name}=={package_version}', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
            else:
                logging.info(f"{package_name} is already installed")
        else:
            if not is_package_installed(package):
                logging.info(f"Installing {package}")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            else:
                logging.info(f"{package} is already installed")

def check_cuda():
    try:
        import torch
        logging.info(f"PyTorch version: {getattr(torch, '__version__', 'unknown')}")
        if not hasattr(torch, 'cuda'):
            logging.error("CUDA attribute not found in torch. Ensure CUDA-enabled PyTorch is installed.")
            error_logger.error("CUDA attribute not found in torch. Ensure CUDA-enabled PyTorch is installed.")
            messagebox.showerror("Error", "CUDA attribute not found in torch. Ensure CUDA-enabled PyTorch is installed.")
            sys.exit(1)
        if not torch.cuda.is_available():
            logging.error("CUDA is not available. Ensure you have a compatible GPU and CUDA is properly installed.")
            error_logger.error("CUDA is not available. Ensure you have a compatible GPU and CUDA is properly installed.")
            messagebox.showerror("Error", "CUDA is not available. Ensure you have a compatible GPU and CUDA is properly installed.")
            sys.exit(1)
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    except ImportError as e:
        logging.error(f"Error importing torch: {e}")
        error_logger.error(f"Error importing torch: {e}")
        messagebox.showerror("Error", f"Error importing torch: {e}. Please ensure torch with CUDA support is installed.")
        sys.exit(1)

def extract_audio(video_path, audio_path):
    logging.info(f"Extracting English audio from {video_path}")
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path]
    subprocess.run(command, check=True)
    logging.info(f"Audio extraction completed: {audio_path}")

def diarize_audio(audio_path, diarized_audio_path):
    logging.info(f"Starting speaker diarization for {audio_path}")
    HUGGING_FACE_TOKEN = "your_hugging_face_token"  # Replace with your actual Hugging Face token
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN, device='cuda')
    logging.info("Pipeline initialized")
    diarization = pipeline({"uri": "filename", "audio": audio_path})
    logging.info("Diarization process completed")
    spkrec = sb.pretrained.interfaces.SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_dir", run_opts={"device":"cuda"})
    logging.info("SpeechBrain model loaded")
    with open("diarization.txt", "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(f"{turn.start:.1f} {turn.end:.1f} {speaker}\n")
            logging.info(f"Diarization turn: {turn.start:.1f} {turn.end:.1f} {speaker}")
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_path = f"segment_{speaker}_{int(turn.start)}.wav"
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
    install_packages()
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
