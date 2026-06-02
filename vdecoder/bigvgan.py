import torch
from torch import nn
from torch.nn import functional as F


class BigVGANMelHead(nn.Module):
    """把 VITS latent z 显式解码成 BigVGAN 需要的 128-band log-mel。

    这是真正的 acoustic model:不再靠随机 1x1 卷积去赌分布,而是被 train.py 里
    BigVGAN 自带 mel 定义生成的 target 直接 L1 监督,先把 mel 拉到 vocoder 的输入流形上。
    输入额外拼入 f0(log 域)与 uv,给 head 提供显式音高线索,缓解纯 mel 路径在歌声
    大动态音高下的谐波漂移。speaker embedding g 通过 FiLM 风格的逐层偏置注入。

    用自包含的 dilated 残差卷积栈实现,刻意不依赖 modules.WN —— 避免 vdecoder 在被
    独立导入时触发 modules 包的循环导入(modules.modules <-> modules.attentions)。
    """

    def __init__(self, latent_channels, n_mel_channels=128, hidden_channels=256,
                 kernel_size=5, n_layers=6, gin_channels=0):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_layers = n_layers
        # latent + f0(1) + uv(1)
        self.pre = nn.Conv1d(latent_channels + 2, hidden_channels, 1)
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
        else:
            self.cond = None
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            pad = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                          dilation=dilation, padding=pad)
            )
            self.res_skip_layers.append(nn.Conv1d(hidden_channels, hidden_channels, 1))
        self.proj = nn.Conv1d(hidden_channels, n_mel_channels, 1)
        self.hidden_channels = hidden_channels

    def forward(self, z, f0, g=None):
        # f0: [B, T] -> log 域归一,unvoiced(f0<=0) 置 0
        uv = (f0 > 0).to(z.dtype).unsqueeze(1)
        lf0 = torch.log(f0.clamp_min(1.0)).unsqueeze(1) / 7.0 * uv
        x = self.pre(torch.cat([z, lf0, uv], dim=1))
        g_cond = self.cond(g) if (self.cond is not None and g is not None) else None
        n = self.hidden_channels
        for i in range(self.n_layers):
            h = self.in_layers[i](x)
            if g_cond is not None:
                h = h + g_cond[:, i * 2 * n:(i + 1) * 2 * n, :]
            acts = torch.tanh(h[:, :n, :]) * torch.sigmoid(h[:, n:, :])
            x = x + self.res_skip_layers[i](acts)
        return self.proj(x)


class BigVGANDecoder(nn.Module):
    """两段式:BigVGANMelHead(z -> 128band mel) + 冻结的官方 BigVGAN vocoder(mel -> 波形)。

    forward 返回 (audio, pred_mel);pred_mel 交给训练循环做直接 mel 监督。
    vocoder 加载官方 universal 权重并冻结,只训 mel head 与 VITS 主干。
    """

    def __init__(
        self,
        latent_channels,
        n_mel_channels=128,
        model_name="nvidia/bigvgan_v2_44khz_128band_512x",
        trainable=False,
        use_cuda_kernel=False,
        remove_weight_norm=True,
        gin_channels=0,
    ):
        super().__init__()
        try:
            import bigvgan
        except ImportError as exc:
            raise ImportError(
                "vocoder_name='bigvgan-v2' requires BigVGAN. Install it with `pip install bigvgan>=2.4.1`."
            ) from exc

        self.mel_head = BigVGANMelHead(
            latent_channels, n_mel_channels=n_mel_channels, gin_channels=gin_channels
        )
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
        pred_mel = self.mel_head(z, f0, g=g)
        if hasattr(self.vocoder, "decode"):
            audio = self.vocoder.decode(pred_mel)
        else:
            audio = self.vocoder(pred_mel)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio, pred_mel
