import os

import torch

from vencoder.encoder import SpeechEncoder
from vencoder.whisper.audio import log_mel_spectrogram, pad_or_trim
from vencoder.whisper.model import ModelDimensions, Whisper


class WhisperPPGLarge(SpeechEncoder):
    def __init__(self, vec_path=None, device=None):
        super().__init__()
        if vec_path is None:
            # 优先用 large-v3(128 mel)，回退 large-v2(80 mel)
            for cand in ("pretrain/large-v3.pt", "pretrain/large-v2.pt"):
                if os.path.exists(cand):
                    vec_path = cand
                    break
            vec_path = vec_path or "pretrain/large-v2.pt"
        if device is None:
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dev = torch.device(device)
        checkpoint = torch.load(vec_path, map_location=device)
        dims = ModelDimensions(**checkpoint["dims"])
        model = Whisper(dims)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.n_mels = dims.n_mels  # 80(v2) 或 128(v3)，决定 mel 提取
        self.hidden_dim = dims
        self.model = model.to(self.dev)

    def encoder(self, wav):
        audio = wav
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio, self.n_mels).to(self.dev)
        with torch.no_grad():
            ppg = self.model.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = torch.FloatTensor(ppg[:ppgln, ]).to(self.dev)
            return ppg[None, :, :].transpose(1, 2)
