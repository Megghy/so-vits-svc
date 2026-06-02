# MS-SB-CQT discriminator ported from NVIDIA/BigVGAN (MIT License).
# Multi-scale sub-band constant-Q transform discriminator.

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torchaudio.transforms import Resample, Spectrogram


def get_2d_padding(kernel_size: Tuple[int, int], dilation: Tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg: dict, hop_length: int, n_octaves: int, bins_per_octave: int):
        super().__init__()
        self.cfg = cfg

        self.filters = cfg["cqtd_filters"]
        self.max_filters = cfg["cqtd_max_filters"]
        self.filters_scale = cfg["cqtd_filters_scale"]
        self.kernel_size = (3, 9)
        self.dilations = cfg["cqtd_dilations"]
        self.stride = (1, 2)

        self.in_channels = cfg["cqtd_in_channels"]
        self.out_channels = cfg["cqtd_out_channels"]
        self.fs = cfg["sampling_rate"]
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                )
            )
        )

        self.conv_post = weight_norm(
            nn.Conv2d(
                out_chs,
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = self.cfg.get("cqtd_normalize_volume", False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        if self.cqtd_normalize_volume:
            x = x - x.mean(dim=-1, keepdims=True)
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)
        # cuFFT 不支持 fp16/bf16，CQT 变换强制走 fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = self.cqt_transform(x.float())

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)
        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[:, :, :, i * self.bins_per_octave: (i + 1) * self.bins_per_octave]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for l in self.convs:
            latent_z = l(latent_z)
            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)
        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.cfg.setdefault("cqtd_filters", 32)
        self.cfg.setdefault("cqtd_max_filters", 1024)
        self.cfg.setdefault("cqtd_filters_scale", 1)
        self.cfg.setdefault("cqtd_dilations", [1, 2, 4])
        self.cfg.setdefault("cqtd_in_channels", 1)
        self.cfg.setdefault("cqtd_out_channels", 1)
        self.cfg.setdefault("cqtd_hop_lengths", [512, 256, 256])
        self.cfg.setdefault("cqtd_n_octaves", [9, 9, 9])
        self.cfg.setdefault("cqtd_bins_per_octaves", [24, 36, 48])

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    self.cfg,
                    hop_length=self.cfg["cqtd_hop_lengths"][i],
                    n_octaves=self.cfg["cqtd_n_octaves"][i],
                    bins_per_octave=self.cfg["cqtd_bins_per_octaves"][i],
                )
                for i in range(len(self.cfg["cqtd_hop_lengths"]))
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, resolution, channel_mult=1, use_spectral_norm=False):
        super().__init__()
        self.resolution = resolution
        self.lrelu_slope = 0.1
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        d = channel_mult
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, int(32 * d), (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32 * d), int(32 * d), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32 * d), int(32 * d), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32 * d), int(32 * d), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32 * d), int(32 * d), (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(int(32 * d), 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode="reflect")
        x = x.squeeze(1)
        # cuFFT 不支持 fp16/bf16，STFT 强制走 fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            window = torch.hann_window(win_length, device=x.device, dtype=torch.float32)
            x = torch.stft(x.float(), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                           window=window, center=False, return_complex=True)
        x = torch.view_as_real(x)
        return torch.norm(x, p=2, dim=-1)  # [B, F, TT]

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        resolutions = cfg.get("resolutions", [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]])
        channel_mult = cfg.get("discriminator_channel_mult", 1)
        use_spectral_norm = cfg.get("mrd_use_spectral_norm", False)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(r, channel_mult, use_spectral_norm) for r in resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorB(nn.Module):
    def __init__(self, window_length, channels=32, hop_factor=0.25,
                 bands=((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))):
        super().__init__()
        self.window_length = window_length
        self.spec_fn = Spectrogram(
            n_fft=window_length, hop_length=int(window_length * hop_factor),
            win_length=window_length, power=None,
        )
        n_fft = window_length // 2 + 1
        self.bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        convs = lambda: nn.ModuleList([
            weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
        ])
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x):
        x = x - x.mean(dim=-1, keepdims=True)
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        # cuFFT 不支持 fp16/bf16，Spectrogram 强制走 fp32
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = self.spec_fn(x.float())
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 2, 1)  # [B, F, T, C] -> [B, C, T, F]
        return [x[..., b[0]: b[1]] for b in self.bands]

    def forward(self, x):
        x_bands = self.spectrogram(x.squeeze(1))
        fmap = []
        out = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = F.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            out.append(band)
        out = torch.cat(out, dim=-1)
        out = self.conv_post(out)
        fmap.append(out)
        return out, fmap


class MultiBandDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        fft_sizes = cfg.get("mbd_fft_sizes", [2048, 1024, 512])
        self.discriminators = nn.ModuleList([DiscriminatorB(window_length=w) for w in fft_sizes])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class CombinedDiscriminator(nn.Module):
    """Chains multiple discriminators, concatenating their outputs into the
    flat 4-list interface that the training loop and losses expect.

    kinds 与 discriminators 一一对应(如 ["mpd","cqt","mrd"]);forward 的 active
    传一个 kind 集合时只跑其中的子判别器,用于训练早期只启用 MPD、按 step 再接入
    CQT/MRD,避免随机初始化的 mel head 阶段被多判别器对抗梯度带偏。"""

    def __init__(self, discriminators: List[nn.Module], kinds: List[str] = None):
        super().__init__()
        self.discriminators = nn.ModuleList(discriminators)
        self.kinds = kinds if kinds is not None else [None] * len(discriminators)

    def forward(self, y, y_hat, active=None):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for kind, disc in zip(self.kinds, self.discriminators):
            if active is not None and kind not in active:
                continue
            y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)
            y_d_rs.extend(y_d_r)
            y_d_gs.extend(y_d_g)
            fmap_rs.extend(fmap_r)
            fmap_gs.extend(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
