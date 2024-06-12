import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import os

class CustomEncoderWav2vec2Classifier:
    def __init__(self, source=None, hparams_file=None, savedir=None, run_opts=None, overrides=None):
        if not source:
            source = r"D:\Python Programs\AutoAudioExtractor\pretrained_models\spkrec-ecapa-voxceleb"
        self.classifier = EncoderClassifier.from_hparams(source=source, hparams_file=hparams_file, savedir=savedir, run_opts=run_opts, overrides=overrides)
    
    def encode_batch(self, wavs):
        embeddings = self.classifier.encode_batch(wavs)
        return embeddings
    
    def classify_batch(self, wavs):
        predictions = self.classifier.classify_batch(wavs)
        return predictions

    def classify_file(self, file_path):
        predictions = self.classifier.classify_file(file_path)
        return predictions
