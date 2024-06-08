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

# Check if requirements.txt exists
if not os.path.exists('requirements.txt'):
    logging.error("requirements.txt file not found. Exiting...")
    sys.exit(1)

# Function to install necessary packages from requirements.txt
def install_packages():
    try:
        logging.info("Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        logging.info("Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logging.info("Packages installed successfully from requirements.txt")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing packages: {e}")
        logging.info("Attempting to install packages individually...")
        try:
            with open('requirements.txt', 'r') as f:
                for line in f:
                    package = line.strip()
                    if package:
                        logging.info(f"Installing {package}...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logging.info("Individual package installation completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing individual packages: {e}")
            sys.exit(1)

# Install packages
install_packages()

# Now import the installed packages
import numpy as np
from pyannote.audio import Pipeline, Model
from pydub import AudioSegment
import psutil
import speechbrain as sb

# Your Hugging Face API token
HUGGING_FACE_TOKEN = "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"

# Load the segmentation model
try:
    logging.info("Loading segmentation model...")
    model = Model.from_pretrained("pyannote/segmentation", use_auth_token=HUGGING_FACE_TOKEN)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    sys.exit(1)

# Ensure ffmpeg is in PATH or set FFMPEG_BINARY environment variable
os.environ["FFMPEG_BINARY"] = "ffmpeg"  # Set this to the path to ffmpeg if not in PATH

# Function to check the audio streams in the video file using ffmpeg
def check_audio_streams(video_path):
    result = subprocess.run(['ffmpeg', '-i', video_path], stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    stream_info = result.stderr
    audio_streams = [line for line in stream_info.split('\n') if 'Audio' in line and 'eng' in line]
    return audio_streams

# Function to extract English audio from video using pydub
def extract_audio(video_path, audio_path):
    logging.info("Extracting English audio from the video...")
    audio_streams = check_audio_streams(video_path)
    
    if not audio_streams:
        raise ValueError("No English audio streams found in the MKV file.")
    
    # Convert the audio to WAV format using pydub
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        logging.info("Audio extraction completed.")
    except Exception as e:
        logging.error(f"Error during audio extraction: {e}")
        sys.exit(1)

# Function to perform speaker diarization using pyannote.audio and speechbrain
def diarize_audio(audio_path, diarized_audio_path):
    logging.info("Performing speaker diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN)
    diarization = pipeline({"uri": "filename", "audio": audio_path})

    # Load the speechbrain model
    spkrec = sb.pretrained.interfaces.SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_dir")

    with open("diarization.txt", "w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            f.write(f"{turn.start:.1f} {turn.end:.1f} {speaker}\n")

    # Create segments for each speaker
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment_path = f"segment_{speaker}_{int(turn.start)}.wav"
        command = ['ffmpeg', '-i', audio_path, '-ss', str(turn.start), '-to', str(turn.end), '-c', 'copy', segment_path]
        subprocess.call(command)
        segments.append(segment_path)

    # Combine segments into one file
    with open("segments_list.txt", "w") as f:
        for segment in segments:
            f.write(f"file '{os.path.abspath(segment)}'\n")

    command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'segments_list.txt', '-c', 'copy', diarized_audio_path]
    subprocess.call(command)
    logging.info("Speaker diarization completed.")

def prompt_for_diarization():
    prompt_root = tk.Toplevel()
    prompt_root.withdraw()
    prompt_root.title("Proceed with Diarization")
    prompt_root.attributes('-topmost', True)
    result = messagebox.askyesno("Proceed with Diarization", "Do you want to proceed with automatic diarization?", parent=prompt_root)
    prompt_root.destroy()
    return result

# Main script
root = tk.Tk()
root.withdraw()
root.lift()
root.attributes('-topmost', True)

video_path = filedialog.askopenfilename(title="Select MKV File", filetypes=[("MKV files", "*.mkv")])
root.attributes('-topmost', False)
if not video_path:
    logging.error("No file selected. Exiting...")
    sys.exit()

logging.info(f"Selected file: {video_path}")

# Determine the best location for the temporary directory on a local device
temp_dir_base = tempfile.gettempdir()

logging.info(f"Creating temporary directory in: {temp_dir_base}")

# Create a temporary directory for processing
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
        messagebox.showerror("Error", str(e))
        sys.exit(1)

    if prompt_for_diarization():
        try:
            diarize_audio(audio_path, diarized_audio_path)
            messagebox.showinfo("Success", f"Diarized audio saved as {diarized_audio_path}")
        except Exception as e:
            logging.error(f"Diarization error: {e}")
            messagebox.showerror("Diarization Error", str(e))
            sys.exit(1)
    else:
        messagebox.showinfo("Cancelled", "Diarization cancelled")

# Ensure all temporary files are deleted
try:
    shutil.rmtree(temp_dir)
    logging.info("Temporary files deleted.")
except OSError as e:
    logging.error(f"Error deleting temporary files: {e}")

root.destroy()
