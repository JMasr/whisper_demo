import os
import torch
import whisper
import torchaudio
import numpy as np


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, device, split="test-clean"):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=os.path.expanduser("~/.cache"), url=split, download=True)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)

        return (mel, text)


class StandarDataset(torch.utils.data.Dataset):
    """
    A simple class to wrap any set of audios and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, audio_path, transcription_path, language: str, resampling_rate: int = 16e3):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resampling_rate = resampling_rate
        self.dataset = self._load_pairs_audio_trans(audio_path, transcription_path)
        self.language = language

    @staticmethod
    def _compute_SAD(sig, fs, threshold=0.0001, sad_start_end_sil_length=100, sad_margin_length=50):
        """ Compute threshold based sound activity """
        # Leading/Trailing margin
        sad_start_end_sil_length = int(sad_start_end_sil_length * 1e-3 * fs)
        # Margin around active samples
        sad_margin_length = int(sad_margin_length * 1e-3 * fs)

        sample_activity = np.zeros(sig.shape)
        sample_activity[np.power(sig, 2) > threshold] = 1
        sad = np.zeros(sig.shape)
        for i in range(sample_activity.shape[1]):
            if sample_activity[0, i] == 1: sad[0, i - sad_margin_length:i + sad_margin_length] = 1
        sad[0, 0:sad_start_end_sil_length] = 0
        sad[0, -sad_start_end_sil_length:] = 0
        return sad

    def _read_audio(self, filepath: str):
        """ This code does the following:
        1. Read audio,
        2. Resample the audio if required,
        3. Perform waveform normalization,
        4. Compute sound activity using threshold based method
        5. Discard the silence regions

        :type filepath: str Specify the path to the file.
        :return a nummpy array with the audio samples and the samplerate.
        """
        # Reading
        s, fs = torchaudio.load(filepath)
        # Resampling
        if fs != self.resampling_rate:
            s, fs = torchaudio.sox_effects.apply_effects_tensor(s, fs, [['rate', str(self.resampling_rate)]])
        # Normalization
        if s.shape[0] > 1:
            s = s.mean(dim=0).unsqueeze(0)
        s = s / torch.max(torch.abs(s))
        s = s / torch.max(torch.abs(s))
        # SAD
        sad = self._compute_SAD(s.numpy(), self.resampling_rate)
        s = s[np.where(sad == 1)]
        return s, fs

    def _load_pairs_audio_trans(self, audio_path, transcription_path):
        dataset = []
        audio_files = sorted(os.listdir(audio_path))
        trans_files = sorted(os.listdir(transcription_path))

        for audio_name, trans_name in zip(audio_files, trans_files):
            with open(os.path.join(transcription_path, trans_name)) as file:
                trans = file.read()

            audio_full_path = os.path.join(audio_path, audio_name)
            audio = whisper.load_audio(audio_full_path, self.resampling_rate)
            dataset.append([audio, self.resampling_rate, trans, f'{os.getcwd()}/{audio_full_path}'])
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, path = self.dataset[item]
        assert sample_rate == self.resampling_rate
        audio_trim = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_trim).to(self.device)

        return mel, text, path
