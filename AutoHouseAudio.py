import os
import sys
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog
import wave

# Ensure ffmpeg is in PATH
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    print("ffmpeg is not installed or not in PATH. Please install ffmpeg and try again.")
    sys.exit(1)

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    command = ['ffmpeg', '-i', video_path, '-map', '0:2', '-q:a', '0', audio_path]
    subprocess.run(command, check=True)

def diarize_audio(audio_path, diarized_audio_path):
    from pyAudioAnalysis import audioSegmentation as aS

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

def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    video_path = filedialog.askopenfilename(title="Select MKV File", filetypes=[("MKV files", "*.mkv")])
    if not video_path:
        print("No file selected. Exiting...")
        sys.exit()
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    diarized_audio_path = os.path.splitext(video_path)[0] + "_diarized.wav"
    extract_audio(video_path, audio_path)
    diarize_audio(audio_path, diarized_audio_path)
    print(f"Diarized audio saved as {diarized_audio_path}")

if __name__ == "__main__":
    main()
