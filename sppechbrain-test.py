import logging
import torch
from pyannote.audio import Pipeline
import speechbrain as sb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_cuda():
    logging.info("Checking CUDA availability")
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. Ensure you have a compatible GPU and CUDA is properly installed.")
        return False
    logging.info("CUDA is available")
    return True

def test_pyannote_pipeline(audio_path):
    logging.info("Testing pyannote Pipeline")
    try:
        HUGGING_FACE_TOKEN = "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"
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN, device='cuda')
        logging.info("Pipeline initialized")
        diarization = pipeline({"uri": "filename", "audio": audio_path})
        logging.info("Diarization process completed")
        return True
    except Exception as e:
        logging.error(f"Error in pyannote pipeline: {e}")
        return False

def test_speechbrain():
    logging.info("Testing SpeechBrain model loading")
    try:
        spkrec = sb.pretrained.interfaces.SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_dir", run_opts={"device":"cuda"})
        logging.info("SpeechBrain model loaded")
        return True
    except Exception as e:
        logging.error(f"Error loading SpeechBrain model: {e}")
        return False

def main():
    logging.info("Starting test")
    audio_path = "path_to_a_test_audio_file.wav"  # Replace with the path to a test audio file

    if test_cuda():
        if test_pyannote_pipeline(audio_path) and test_speechbrain():
            logging.info("All tests passed successfully")
        else:
            logging.error("Some tests failed")
    else:
        logging.error("CUDA test failed")

if __name__ == "__main__":
    main()
