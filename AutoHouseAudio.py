import logging
import os
import subprocess
import sys
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a custom logger to also log errors to a file
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler('error-log.txt')
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(error_formatter)
error_logger.addHandler(error_handler)

# Ensure ffmpeg is in PATH or set FFMPEG_BINARY environment variable
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    logging.info(f"ffmpeg is installed at: {ffmpeg_path}")
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
else:
    logging.error("ffmpeg is not installed or not in PATH")
    error_logger.error("ffmpeg is not installed or not in PATH")
    messagebox.showerror("Error", "ffmpeg is not installed or not in PATH. Please install ffmpeg and try again.")
    sys.exit(1)

# Determine virtual environment directory
venv_root_dir = os.environ.get('VENV_DIR', os.path.join(os.path.expanduser("~"), 'venv'))
venv_path = os.path.join(venv_root_dir, 'autohouse_audio_env')
if not os.path.exists(venv_path):
    logging.info(f"Creating virtual environment at: {venv_path}")
    subprocess.check_call([sys.executable, '-m', 'venv', venv_path])
else:
    logging.info(f"Virtual environment already exists at: {venv_path}")

# Install pydub and CUDA-based pyannote.audio in the virtual environment
if os.name == 'nt':
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'torch==2.0.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'torchaudio==2.0.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'torchvision==2.0.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
else:
    python_executable = os.path.join(venv_path, 'bin', 'python')

try:
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'pydub'])
    # Install CUDA version of torch explicitly for CUDA 11.7
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'torch==1.11.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'torchaudio==0.11.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'torchvision==0.12.0+cu117', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
    subprocess.check_call([python_executable, '-m', 'pip', 'install', 'pyannote.audio[cuda]'])
except subprocess.CalledProcessError as e:
    logging.error(f"Failed to install packages: {e}")
    error_logger.error(f"Failed to install packages: {e}")
    messagebox.showerror("Error", "Failed to install required packages in the virtual environment. Please check your installation.")
    sys.exit(1)

# Re-run the script in the context of the virtual environment
if sys.executable != python_executable:
    logging.info(f"Re-running script with virtual environment: {python_executable}")
    result = subprocess.run([python_executable] + sys.argv)
    sys.exit(result.returncode)

# At this point, the script is running in the virtual environment with pydub and pyannote.audio installed
try:
    from pydub import AudioSegment
    from pyannote.audio import Pipeline
    import torch
    import speechbrain as sb
    logging.info("pydub, pyannote.audio, torch, and speechbrain libraries loaded successfully in the virtual environment")
except ImportError as e:
    logging.error(f"Error importing pydub, pyannote.audio, torch, or speechbrain: {e}")
    error_logger.error(f"Error importing pydub, pyannote.audio, torch, or speechbrain: {e}")
    messagebox.showerror("Error", f"Error importing pydub, pyannote.audio, torch, or speechbrain: {e}. Please ensure they are installed.")
    sys.exit(1)

# Check CUDA availability
logging.info(f"PyTorch CUDA is available: {torch.cuda.is_available()}")
logging.info(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
logging.info(f"PyTorch CUDA current device: {torch.cuda.current_device()}")
logging.info(f"PyTorch CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Function to check the audio streams in the video file using ffmpeg
def check_audio_streams(video_path):
    logging.info(f"Checking audio streams for {video_path}")
    result = subprocess.run(['ffmpeg', '-i', video_path], stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    stream_info = result.stderr
    audio_streams = [line for line in stream_info.split('\n') if 'Audio' in line and 'eng' in line]
    logging.info(f"Audio streams found: {audio_streams}")
    return audio_streams

# Function to extract English audio from video using pydub
def extract_audio(video_path, audio_path):
    logging.info(f"Extracting English audio from {video_path}")
    audio_streams = check_audio_streams(video_path)
    if not audio_streams:
        logging.error("No English audio streams found in the MKV file.")
        error_logger.error("No English audio streams found in the MKV file.")
        raise ValueError("No English audio streams found in the MKV file.")
    
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        logging.info(f"Audio extraction completed: {audio_path}")
    except Exception as e:
        logging.error(f"Error during audio extraction: {e}")
        error_logger.error(f"Error during audio extraction: {e}")
        messagebox.showerror("Error", f"Error during audio extraction: {e}")
        sys.exit(1)

# Function to perform speaker diarization using pyannote.audio and speechbrain
def diarize_audio(audio_path, diarized_audio_path):
    logging.info(f"Starting speaker diarization for {audio_path}")
    try:
        logging.info("Checking CUDA availability")
        if not torch.cuda.is_available():
            logging.error("CUDA is not available. Ensure you have a compatible GPU and CUDA is properly installed.")
            error_logger.error("CUDA is not available. Ensure you have a compatible GPU and CUDA is properly installed.")
            messagebox.showerror("Error", "CUDA is not available. Ensure you have a compatible GPU and CUDA is properly installed.")
            sys.exit(1)
        logging.info("CUDA is available")

        logging.info("Initializing pyannote Pipeline")
        HUGGING_FACE_TOKEN = "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"
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN, device='cuda')
        logging.info("Pipeline initialized")

        logging.info("Starting diarization process")
        diarization = pipeline({"uri": "filename", "audio": audio_path})
        logging.info("Diarization process completed")

        logging.info("Loading SpeechBrain model")
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
            logging.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                logging.error(f"Error creating segment: {segment_path}, Error: {result.stderr}")
                error_logger.error(f"Error creating segment: {segment_path}, Error: {result.stderr}")
            else:
                logging.info(f"Created segment: {segment_path}")
            segments.append(segment_path)

        with open("segments_list.txt", "w") as f:
            for segment in segments:
                f.write(f"file '{os.path.abspath(segment)}'\n")
                logging.info(f"Segment added to list: {segment}")

        logging.info("Merging segments into final diarized audio file...")
        command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'segments_list.txt', '-c', 'copy', diarized_audio_path]
        logging.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logging.error(f"Error in merging segments: {result.stderr}")
            error_logger.error(f"Error in merging segments: {result.stderr}")
            messagebox.showerror("Error", "Error in merging segments to create the final diarized audio file.")
            raise RuntimeError("Error in merging segments to create the final diarized audio file.")
        else:
            logging.info("Speaker diarization completed successfully.")
    except ImportError as e:
        logging.error(f"Error during speaker diarization: {e}")
        error_logger.error(f"Error during speaker diarization: {e}")
        logging.error(f"Current PYTHONPATH: {sys.path}")
        messagebox.showerror("Error", f"Error during speaker diarization: {e}. Please ensure all required packages are installed.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error in speaker diarization: {e}")
        error_logger.error(f"Error in speaker diarization: {e}")
        messagebox.showerror("Error", f"Error in speaker diarization: {e}")
        sys.exit(1)

def prompt_for_diarization():
    logging.info("Prompting user for diarization confirmation")
    prompt_root = tk.Toplevel()
    prompt_root.withdraw()
    prompt_root.title("Proceed with Diarization")
    prompt_root.attributes('-topmost', True)
    result = messagebox.askyesno("Proceed with Diarization", "Do you want to proceed with automatic diarization?", parent=prompt_root)
    prompt_root.destroy()
    logging.info(f"User response for diarization: {result}")
    return result

def main():
    logging.info("Starting main function")
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)

    video_path = filedialog.askopenfilename(title="Select MKV File", filetypes=[("MKV files", "*.mkv")])
    root.attributes('-topmost', False)
    if not video_path:
        logging.error("No file selected. Exiting...")
        error_logger.error("No file selected. Exiting...")
        messagebox.showerror("Error", "No file selected. Exiting...")
        sys.exit()

    logging.info(f"Selected file: {video_path}")

    temp_dir_base = tempfile.gettempdir()
    logging.info(f"Creating temporary directory in: {temp_dir_base}")

    with tempfile.TemporaryDirectory(dir=temp_dir_base) as temp_dir:
        temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
        logging.info(f"Copying video file to temporary directory: {temp_video_path}")
        shutil.copy(video_path, temp_video_path)
        
        audio_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
        diarized_audio_path = os.path.splitext(video_path)[0] + "_diarized.wav"

        try:
            extract_audio(temp_video_path, audio_path)
        except ValueError as e:
            logging.error(f"Error extracting audio: {e}")
            error_logger.error(f"Error extracting audio: {e}")
            messagebox.showerror("Error", str(e))
            sys.exit(1)

        if prompt_for_diarization():
            try:
                diarize_audio(audio_path, diarized_audio_path)
                messagebox.showinfo("Success", f"Diarized audio saved as {diarized_audio_path}")
            except Exception as e:
                logging.error(f"Diarization error: {e}")
                error_logger.error(f"Diarization error: {e}")
                messagebox.showerror("Diarization Error", str(e))
                sys.exit(1)
        else:
            logging.info("Diarization cancelled by user")
            messagebox.showinfo("Cancelled", "Diarization cancelled")

    root.destroy()
    logging.info("Main function completed")

    # Clean up: delete the virtual environment and other temporary files
    try:
        if os.path.exists(venv_path):
            shutil.rmtree(venv_path)
            logging.info(f"Deleted virtual environment at: {venv_path}")
        else:
            logging.info(f"No virtual environment found at: {venv_path}")

        # Optionally, delete any other temp files created
        temp_files = ["diarization.txt", "segments_list.txt"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logging.info(f"Deleted temporary file: {temp_file}")
            else:
                logging.info(f"No temporary file found: {temp_file}")

    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        error_logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
