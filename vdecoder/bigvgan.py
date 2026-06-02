import torch
from torch import nn


class BigVGANDecoder(nn.Module):
    def __init__(
        self,
        latent_channels,
        n_mel_channels=128,
        model_name="nvidia/bigvgan_v2_44khz_128band_512x",
        trainable=False,
        use_cuda_kernel=False,
        remove_weight_norm=True,
    ):
        super().__init__()
        try:
            import bigvgan
        except ImportError as exc:
            raise ImportError(
                "vocoder_name='bigvgan-v2' requires BigVGAN. Install it with `pip install bigvgan>=2.4.1`."
            ) from exc

        self.mel_proj = nn.Conv1d(latent_channels, n_mel_channels, 1)
        self.vocoder = bigvgan.BigVGAN._from_pretrained(
            model_id=model_name,
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            token=None,
            use_cuda_kernel=use_cuda_kernel,
        )
        self.trainable = trainable
        if remove_weight_norm and hasattr(self.vocoder, "remove_weight_norm"):
            self.vocoder.remove_weight_norm()
        for param in self.vocoder.parameters():
            param.requires_grad = trainable
        if not trainable:
            self.vocoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.vocoder.eval()
        return self

    def forward(self, z, g=None, f0=None):
        mel = self.mel_proj(z)
        if hasattr(self.vocoder, "decode"):
            audio = self.vocoder.decode(mel)
        else:
            audio = self.vocoder(mel)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio
