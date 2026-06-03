"""NSF-BigVGAN 混合声码器：结合 BigVGAN 的高质量 upsampling 和 NSF 的 f0 激励源。

架构：
    latent -> BigVGANMelHead -> 128-band mel ───┐
                                                 ├─> BigVGAN upsampling
    f0 -> SourceModuleHnNSF -> harmonic source ─┘   (逐层注入)
                                                        ↓
                                                    waveform

关键设计：
1. 复用 BigVGANMelHead 保持 128-band mel 直接监督
2. 复用 SourceModuleHnNSF 生成 f0 正弦激励
3. 每个 upsample stage 后通过 noise_convs 注入 harmonic source
4. 支持分阶段训练（Phase1 冻结 BigVGAN 主干，Phase2 低 lr 解冻）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .bigvgan import BIGVGAN_CACHE_DIR, BigVGANMelHead
from .hifigan.models import SourceModuleHnNSF


class NSFBigVGANGenerator(nn.Module):
    """NSF-BigVGAN 生成器核心：从 mel + f0 生成波形。

    不直接继承官方 BigVGAN，而是组合方式：加载官方权重到内部模块，
    插入 NSF source 和 noise_convs。这样可以：
    - 避免依赖官方 BigVGAN 的内部实现细节
    - 灵活控制哪些层参与训练
    - 方便做权重加载的 shape 验证
    """

    def __init__(
        self,
        model_name: str = "nvidia/bigvgan_v2_44khz_128band_512x",
        sampling_rate: int = 44100,
        harmonic_num: int = 8,
        use_cuda_kernel: bool = False,
        remove_weight_norm: bool = True,
        trainable: bool = False,
    ):
        super().__init__()

        try:
            import bigvgan
        except ImportError as exc:
            raise ImportError(
                "NSF-BigVGAN requires BigVGAN library. Install: pip install bigvgan>=2.4.1"
            ) from exc

        # 加载官方 BigVGAN 作为基础
        self.bigvgan_base = bigvgan.BigVGAN._from_pretrained(
            model_id=model_name,
            revision=None,
            cache_dir=BIGVGAN_CACHE_DIR,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            token=None,
            use_cuda_kernel=use_cuda_kernel,
        )

        if remove_weight_norm and hasattr(self.bigvgan_base, "remove_weight_norm"):
            self.bigvgan_base.remove_weight_norm()

        # 提取 BigVGAN 的 upsample 配置（从模型推断）
        self.ups = self.bigvgan_base.ups
        self.num_upsamples = len(self.ups)

        # 计算总 upsample 倍率
        upsample_rates = self._infer_upsample_rates()
        self.upp = int(np.prod(upsample_rates))

        # NSF 模块：生成 f0 驱动的 harmonic source
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=harmonic_num,
            sine_amp=0.1,
            add_noise_std=0.003,
            voiced_threshod=0,
        )

        # f0 上采样到波形帧率
        self.f0_upsamp = nn.Upsample(scale_factor=self.upp)

        # noise_convs：将 harmonic source 注入到每个 upsample stage
        # 设计参考 vdecoder/hifigan/models.py:334-348
        self.noise_convs = nn.ModuleList()
        for i in range(self.num_upsamples):
            # 计算当前 stage 输出的 channel 数
            # BigVGAN 的 channel 递减：initial / 2^(i+1)
            if hasattr(self.bigvgan_base, 'num_kernels'):
                # 从 resblocks 推断 channel（更可靠）
                c_cur = self._get_stage_channels(i)
            else:
                # fallback：假设标准配置
                c_cur = 512 // (2 ** (i + 1))

            # 计算剩余的 upsample 倍率（用于对齐 har_source 到当前分辨率）
            if i + 1 < self.num_upsamples:
                stride_f0 = int(np.prod(upsample_rates[i + 1:]))
                self.noise_convs.append(
                    nn.Conv1d(
                        1, c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2
                    )
                )
            else:
                # 最后一层，har_source 已经在目标分辨率
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

        # 小初始化：避免随机 source 一开始就破坏 BigVGAN 的分布
        for conv in self.noise_convs:
            nn.init.xavier_uniform_(conv.weight, gain=0.01)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        # 每个 stage 的可学习 source gate：初始为 0，前向时直接相乘。
        # 训练起点 source 注入量严格为 0，模型完全等价官方 BigVGAN，
        # 之后由梯度自行学会按需放大每层 f0 source，避免相位污染。
        self.source_gains = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.num_upsamples)
        ])

        # 最近一次 forward 的 source 监控统计(供 TensorBoard 读取，训练时填充)
        self.source_stats = {}

        self.trainable = trainable
        if not trainable:
            # Phase1: 冻结 BigVGAN 主干
            for param in self.bigvgan_base.parameters():
                param.requires_grad = False
            self.bigvgan_base.eval()

    def _upsample_layers(self, stage):
        if isinstance(stage, (nn.ModuleList, nn.Sequential)):
            return stage
        return (stage,)

    def _infer_upsample_rates(self):
        """从 BigVGAN upsample stages 推断每一层倍率。"""
        rates = []
        for stage in self.ups:
            stage_rate = 1
            for layer in self._upsample_layers(stage):
                stride = getattr(layer, "stride", None)
                if stride is None:
                    continue
                stage_rate *= int(stride[0] if isinstance(stride, (tuple, list)) else stride)
            if stage_rate == 1:
                raise RuntimeError("Cannot infer BigVGAN upsample rate from stage without stride")
            rates.append(stage_rate)
        return rates

    def _get_stage_channels(self, stage_idx):
        """推断指定 upsample stage 的输出 channel 数。"""
        for layer in self._upsample_layers(self.ups[stage_idx]):
            out_channels = getattr(layer, "out_channels", None)
            if out_channels is not None:
                return int(out_channels)
        raise RuntimeError("Cannot infer BigVGAN stage channels from upsample layer")

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            # 即使调用 .train()，也保持 BigVGAN 主干在 eval 模式
            self.bigvgan_base.eval()
        return self

    def unfreeze_bigvgan(self):
        """Phase2: 解冻 BigVGAN 主干，进入低 lr 全量微调"""
        if self.trainable:
            return
        self.trainable = True
        for param in self.bigvgan_base.parameters():
            param.requires_grad = True
        self.bigvgan_base.train()

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 128, T] - BigVGAN 格式的 log-mel spectrogram
            f0: [B, T] - 基频序列（Hz，unvoiced 为 0）

        Returns:
            audio: [B, 1, T*upp] - 生成的波形
        """
        # 生成 harmonic source
        f0_up = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T*upp, 1]
        har_source, _, uv = self.m_source(f0_up, upp=self.upp)
        har_source = har_source.transpose(1, 2)  # [B, 1, T*upp]

        # source 监控统计：harmonic source 整体/清浊音 RMS + 每层 gain
        if self.training:
            with torch.no_grad():
                uv_mask = uv.transpose(1, 2)  # [B, 1, T*upp]
                voiced = uv_mask.sum().clamp_min(1.0)
                unvoiced = (1.0 - uv_mask).sum().clamp_min(1.0)
                src_sq = har_source.float().pow(2)
                self.source_stats = {
                    "source_rms": src_sq.mean().sqrt().item(),
                    "source_rms_voiced": (src_sq * uv_mask).sum().div(voiced).sqrt().item(),
                    "source_rms_unvoiced": (src_sq * (1.0 - uv_mask)).sum().div(unvoiced).sqrt().item(),
                    "voiced_ratio": (uv_mask.mean()).item(),
                }
                for i, gain in enumerate(self.source_gains):
                    self.source_stats[f"source_gain_{i}"] = gain.item()

        # BigVGAN 主干 forward，手动注入 har_source
        x = self.bigvgan_base.conv_pre(mel)

        num_kernels = getattr(self.bigvgan_base, 'num_kernels', 3)
        resblocks = self.bigvgan_base.resblocks

        for i in range(self.num_upsamples):
            for up_layer in self._upsample_layers(self.ups[i]):
                x = up_layer(x)

            # 注入 harmonic source，按可学习 gate 缩放(初始 gain=0 → 不注入)
            x_source = self.noise_convs[i](har_source)
            x = x + self.source_gains[i] * x_source

            # ResBlocks
            xs = None
            for j in range(num_kernels):
                block_idx = i * num_kernels + j
                if block_idx < len(resblocks):
                    if xs is None:
                        xs = resblocks[block_idx](x)
                    else:
                        xs = xs + resblocks[block_idx](x)

            x = xs / num_kernels

        # 后处理
        x = self.bigvgan_base.activation_post(x)
        audio = self.bigvgan_base.conv_post(x)
        if getattr(self.bigvgan_base, "use_tanh_at_final", True):
            audio = torch.tanh(audio)
        else:
            audio = torch.clamp(audio, min=-1.0, max=1.0)

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        return audio


class NSFBigVGANDecoder(nn.Module):
    """完整的 NSF-BigVGAN 解码器：MelHead + NSFBigVGANGenerator。

    两段式架构：
    1. BigVGANMelHead: latent z -> 128-band mel（直接监督）
    2. NSFBigVGANGenerator: mel + f0 -> waveform（NSF 注入）

    训练策略：
    - Phase1: 冻结 BigVGAN 主干，训练 mel_head + noise_convs + m_source
    - Phase2: 低 lr 解冻 BigVGAN 主干，全量微调
    """

    def __init__(
        self,
        latent_channels: int,
        n_mel_channels: int = 128,
        model_name: str = "nvidia/bigvgan_v2_44khz_128band_512x",
        sampling_rate: int = 44100,
        harmonic_num: int = 8,
        trainable: bool = False,
        use_cuda_kernel: bool = False,
        remove_weight_norm: bool = True,
        gin_channels: int = 0,
    ):
        super().__init__()

        # MelHead: 把 VITS latent 解码成 BigVGAN 需要的 128-band mel
        self.mel_head = BigVGANMelHead(
            latent_channels,
            n_mel_channels=n_mel_channels,
            gin_channels=gin_channels
        )

        # NSF-BigVGAN Generator
        self.vocoder = NSFBigVGANGenerator(
            model_name=model_name,
            sampling_rate=sampling_rate,
            harmonic_num=harmonic_num,
            use_cuda_kernel=use_cuda_kernel,
            remove_weight_norm=remove_weight_norm,
            trainable=trainable,
        )

        self.trainable = trainable

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.vocoder.bigvgan_base.eval()
        return self

    def unfreeze_vocoder(self):
        """运行时解冻声码器，供自动分阶段训练调用（兼容现有接口）"""
        if self.trainable:
            return
        self.trainable = True
        self.vocoder.unfreeze_bigvgan()

    def forward(
        self,
        z: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None
    ):
        """
        Args:
            z: [B, latent_channels, T] - VITS latent
            g: [B, gin_channels, 1] - speaker embedding (optional)
            f0: [B, T] - 基频序列（必需）

        Returns:
            audio: [B, 1, T*upp]
            pred_mel: [B, 128, T] - 预测的 mel（用于直接监督）
        """
        if f0 is None:
            raise ValueError("NSF-BigVGAN requires f0 input")

        # Stage 1: latent -> mel
        pred_mel = self.mel_head(z, f0, g=g)

        # Stage 2: mel + f0 -> audio
        audio = self.vocoder(pred_mel, f0)

        return audio, pred_mel


def load_bigvgan_weights_safely(nsf_bigvgan: NSFBigVGANGenerator, official_ckpt_path: str):
    """安全加载官方 BigVGAN 权重，跳过 shape 不匹配或缺失的层。

    Args:
        nsf_bigvgan: NSFBigVGANGenerator 实例
        official_ckpt_path: 官方 BigVGAN checkpoint 路径

    Returns:
        dict: 加载统计信息 {matched: int, skipped: list}
    """
    official_state = torch.load(official_ckpt_path, map_location='cpu')
    if 'generator' in official_state:
        official_state = official_state['generator']

    nsf_state = nsf_bigvgan.state_dict()

    matched = []
    skipped = []

    for k, v in official_state.items():
        # 官方权重的 key 可能带 'bigvgan_base.' 前缀或不带
        target_key = f'bigvgan_base.{k}' if f'bigvgan_base.{k}' in nsf_state else k

        if target_key in nsf_state:
            if v.shape == nsf_state[target_key].shape:
                nsf_state[target_key] = v
                matched.append(target_key)
            else:
                skipped.append(f"{target_key}: shape mismatch {v.shape} vs {nsf_state[target_key].shape}")
        else:
            skipped.append(f"{k}: not found in NSF-BigVGAN")

    nsf_bigvgan.load_state_dict(nsf_state)

    print(f"✓ Loaded {len(matched)} layers from official BigVGAN")
    if skipped:
        print(f"⚠ Skipped {len(skipped)} layers (expected for new NSF modules):")
        for s in skipped[:5]:
            print(f"  - {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    # 验证新模块确实是随机初始化的
    nsf_modules = [k for k in nsf_state.keys() if 'm_source' in k or 'noise_convs' in k]
    nsf_loaded = [k for k in matched if 'm_source' in k or 'noise_convs' in k]
    if nsf_loaded:
        print(f"⚠ WARNING: NSF modules should NOT be loaded from official BigVGAN: {nsf_loaded}")

    return {
        'matched': len(matched),
        'skipped': skipped,
        'nsf_modules_count': len(nsf_modules),
    }
