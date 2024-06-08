import torch
from pyannote.audio import Pipeline
import speechbrain as sb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_cuda():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("CUDNN version:", torch.backends.cudnn.version())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

def test_pyannote(audio_path):
    logging.info("Testing pyannote Pipeline")
    try:
        HUGGING_FACE_TOKEN = "your_huggingface_token"
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN, device='cuda')
        diarization = pipeline({"uri": "filename", "audio": audio_path})
        logging.info("Diarization process completed")
    except Exception as e:
        logging.error(f"Error in pyannote pipeline: {e}")

def test_speechbrain():
    logging.info("Testing SpeechBrain model loading")
    try:
        spkrec = sb.pretrained.interfaces.SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_dir", run_opts={"device":"cuda"})
        logging.info("SpeechBrain model loaded")
    except Exception as e:
        logging.error(f"Error loading SpeechBrain model: {e}")

if __name__ == "__main__":
    test_cuda()
    test_pyannote("path_to_test_audio.wav")  # Replace with actual test audio path
    test_speechbrain()
