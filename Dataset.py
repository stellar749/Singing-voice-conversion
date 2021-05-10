import os,sys
import logging
import librosa
import numpy as np

# Load and process the data
class DataProcessing:
    def __init__(self, logger, config):
        self.logger=logger
        self.config=config

        self.mixtures = []
        self.accomps = []
        self.vocals = []
        self.batch_index = 0
        
    def load(self, filedir):
        if os.path.isdir(filedir):
            for root, dirs, files in os.walk(filedir):
                for file in filter(lambda f: f.endswith(".wav"), files):
                    self.logger.debug("Loading song %s",os.path.join(root, file))
                    mixture_dual, _ = librosa.load(os.path.join(root, file), sr=self.config.getint("data", "sample_size"), mono = False)

                    mixture = librosa.to_mono(mixture_dual) * 2 #TODO:为何要乘2
                    accomp = mixture_dual[0, :]
                    vocal = mixture_dual[1, :]

                    self.mixtures.append(mixture)
                    self.accomps.append(accomp)
                    self.vocals.append(vocal)

        else:
            self.logger.critical("Folder %s does not exist!", filedir)
            sys.exit(8)

    def stft(self):

        self.mixtures_stft = []
        self.vocals_stft = []

        self.logger.info("Processing stft for data")
        if len(self.mixtures) == 0:
            self.logger.critical("No mixtures for training found.")
            sys.exit(9)
        for i in range(len(self.mixtures)):
                mixture_stft = librosa.stft(self.mixtures[i], n_fft = self.config.getint("data", "window_size"), hop_length = self.config.getint("data", "hop_length"),window='hann')
                vocal_stft = librosa.stft(self.vocals[i], n_fft = self.config.getint("data", "window_size"), hop_length = self.config.getint("data", "hop_length"),window='hann')
                self.mixtures_stft.append(mixture_stft)
                self.vocals_stft.append(vocal_stft)

    def split(self):

        self.mixtures_split = []
        self.vocals_split = []

        self.logger.info("Splitting data")
        if len(self.mixtures) == 0:
            self.logger.critical("No mixtures for training found.")
            sys.exit(9)
        for i in range(len(self.mixtures_stft)):
            sample_length = self.config.getint("data", "sample_length")
            for x in range (self.mixtures_stft[i].shape[1] // sample_length):
                mixture_split = self.mixtures_stft[i][:,x * sample_length : (x + 1) * sample_length]
                vocal_split = self.vocals_stft[i][:,x * sample_length : (x + 1) * sample_length]
                self.mixtures_split.append(mixture_split)
                self.vocals_split.append(vocal_split)
    
    def get_batch(self,batch):
        mixture_batch = self.mixtures_split[self.batch_index:self.batch_index+batch]
        vocal_batch = self.vocals_split[self.batch_index:self.batch_index+batch]
        mixture_batch.transpose.transpose((0, 2, 1))
        vocal_batch.transpose.transpose((0, 2, 1))
        self.batch_index = self.batch_index+batch
        return mixture_batch,vocal_batch
