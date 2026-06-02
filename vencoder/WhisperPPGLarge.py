import os

import torch

from vencoder.encoder import SpeechEncoder
from vencoder.whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
from vencoder.whisper.model import ModelDimensions, Whisper


class WhisperPPGLarge(SpeechEncoder):
    def __init__(self, vec_path="pretrain/large-v3.pt", device=None):
        super().__init__()
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Whisper checkpoint not found: {vec_path}")
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        checkpoint = torch.load(vec_path, map_location=self.dev)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.n_mels = dims.n_mels  # 80(v2) 或 128(v3)，决定 mel 提取
        self.hidden_dim = dims.n_audio_state
        self.model = model.to(self.dev).eval()

    def _encode_chunk(self, audio):
        mel = log_mel_spectrogram(pad_or_trim(audio), self.n_mels).to(self.dev)
        ppg = self.model.encoder(mel.unsqueeze(0))
        return ppg[:, : audio.shape[0] // 320].float()

    def encoder(self, wav):
        audio = wav.to(self.dev)
        chunks = [audio[i:i + N_SAMPLES] for i in range(0, audio.shape[0], N_SAMPLES)]
        with torch.no_grad():
            ppg = torch.cat([self._encode_chunk(chunk) for chunk in chunks], dim=1)
        return ppg.transpose(1, 2)
