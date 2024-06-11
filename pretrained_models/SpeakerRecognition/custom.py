import torch
import torchaudio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model
import os

class CustomSpeakerRecognition:
    def __init__(self, source=None, hparams_file=None, savedir=None, run_opts=None, overrides=None):
        if not source:
            source = os.path.join(os.getcwd(), "pretrained_models", "speakerrecognition")
        self.model = Model.from_pretrained(source)
        self.pipeline = SpeakerDiarization(segmentation=self.model)
    
    def encode_batch(self, wavs):
        embeddings = self.model(wavs)
        return embeddings
    
    def classify_batch(self, wavs):
        diarization = self.pipeline(wavs)
        return diarization
    
    def classify_file(self, file_path):
        diarization = self.pipeline({'uri': file_path, 'audio': file_path})
        return diarization
