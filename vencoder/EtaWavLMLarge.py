import os

import torch

from vencoder.eta_speaker import SpeakerEmbedder
from vencoder.WavLMLarge import WavLMLarge


class EtaWavLMLarge(WavLMLarge):
    """Eta-WavLM (ACL 2025, arXiv:2505.19273) on top of WavLM-Large content features.

    Removes the speaker-dependent component from WavLM-Large features via a fixed
    linear decomposition: for an utterance with speaker x-vector ``s`` (one vector per
    utterance) and bias term, the speaker-predictable component ``b @ W`` (b = [1, s])
    is a constant offset that is subtracted from every frame. ``W`` is fit once on a
    multi-speaker corpus by ``eta_wavlm_fit.py``.

    Output stays identical in shape/rate to WavLM-Large: [1, 1024, T] at 50fps,
    16kHz input, so ssl_dim/gin/filter remain 1024 (same as ``wavlmlarge``).
    The only added effect is stronger content/speaker disentanglement (less timbre leak).
    """

    def __init__(self, vec_path="pretrain/WavLM-Large.pt",
                 proj_path="pretrain/eta_wavlm_proj.pt",
                 spk_path="microsoft/wavlm-base-plus-sv",
                 device=None, output_layer=6):
        super().__init__(vec_path=vec_path, device=device, output_layer=output_layer)
        if not os.path.exists(proj_path):
            raise FileNotFoundError(
                f"Eta-WavLM projection not found: {proj_path}\n"
                f"Fit it first on a MULTI-SPEAKER wav dir:\n"
                f"  python eta_wavlm_fit.py --in_dir <multi_speaker_wav_dir>"
            )
        ckpt = torch.load(proj_path, map_location="cpu")
        self.W = ckpt["W"].to(self.dev).float()  # [spk_dim+1, hidden_dim]
        self.spk = SpeakerEmbedder(model_path=ckpt.get("spk_path", spk_path), device=self.dev)
        assert self.W.shape[0] == self.spk.dim + 1, \
            f"proj speaker dim {self.W.shape[0] - 1} != embedder dim {self.spk.dim}"
        assert self.W.shape[1] == self.hidden_dim, \
            f"proj hidden dim {self.W.shape[1]} != WavLM dim {self.hidden_dim}"

    def encoder(self, wav):
        feats = super().encoder(wav)  # [1, hidden_dim, T]
        wav16k = wav.mean(-1) if wav.dim() == 2 else wav
        s = self.spk.embed(wav16k).to(self.dev)  # [spk_dim]
        b = torch.cat([torch.ones(1, device=self.dev), s])  # [spk_dim+1]
        corr = (b @ self.W).view(1, self.hidden_dim, 1)  # [1, hidden_dim, 1]
        return feats - corr
