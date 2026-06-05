"""BigVGAN 自动分阶段训练策略"""
import os
import torch

from training_config import BIGVGAN_VOCODERS


def _cfg_get(cfg, key, default):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def is_bigvgan_vocoder(hps):
    model = getattr(hps, "model", None)
    return getattr(model, "vocoder_name", "") in BIGVGAN_VOCODERS


class BigVGANStrategy:
    """管理 BigVGAN 两段式声码器的自动分阶段训练。

    Phase1: 冻结声码器,先让 MelHead 对齐固定目标,判别器延后启动
    Phase2: 解冻声码器微调,开启完整判别器对抗
    """

    def __init__(self, hps, model_dir):
        if not is_bigvgan_vocoder(hps):
            raise ValueError("BigVGANStrategy can only be used with BigVGAN vocoders.")
        self.hps = hps
        self.model_dir = model_dir
        self.phase2_marker = os.path.join(model_dir, ".bigvgan_phase2")

        cfg = getattr(hps.train, "bigvgan_strategy", None)
        if cfg is None:
            self.mode = "frozen"
            self.phase1_disc_start = 90000
            self.phase1_mel_target = 0.35
            self.phase2_vocoder_lr = 1e-5
            self.use_mel_loss = True
            self.use_bigvgan_mel_loss = True
            self.disc_warmup_steps = 0
        else:
            self.mode = _cfg_get(cfg, "mode", "frozen")
            self.phase1_disc_start = _cfg_get(cfg, "phase1_disc_start", 90000)
            self.phase1_mel_target = _cfg_get(cfg, "phase1_mel_target", 0.35)
            self.phase2_vocoder_lr = _cfg_get(cfg, "phase2_vocoder_lr", 1e-5)
            self.use_mel_loss = _cfg_get(cfg, "use_mel_loss", True)
            self.use_bigvgan_mel_loss = _cfg_get(cfg, "use_bigvgan_mel_loss", True)
            # no-GAN warmup：前 N 步彻底关闭所有判别器，只跑 mel+KL+f0+bigvgan_mel，
            # 让 MelHead 和 source gate 在无对抗梯度干扰下先稳定。
            self.disc_warmup_steps = _cfg_get(cfg, "disc_warmup_steps", 0)

        self.in_phase2 = os.path.exists(self.phase2_marker)

    def needs_vocoder_finetune(self):
        """判断是否需要声码器微调(拆分 lr 组)"""
        if self.mode == "frozen":
            return False
        if self.mode == "finetune_from_start":
            return True
        if self.mode == "auto_finetune":
            return self.in_phase2
        return False

    def get_disc_start_step(self):
        """返回判别器启动步数"""
        if self.mode == "auto_finetune" and not self.in_phase2:
            return self.phase1_disc_start
        return getattr(self.hps.train, "disc_start_step", 10000)

    def in_disc_warmup(self, global_step):
        """no-GAN warmup 窗口内：彻底跳过判别器前向与对抗/FM 梯度。"""
        return global_step < self.disc_warmup_steps

    def should_transition_to_phase2(self, global_step, recent_mel_losses):
        """检查是否该切换到 Phase2(仅 auto_finetune 模式)"""
        if self.mode != "auto_finetune":
            return False
        if self.in_phase2:
            return False
        if global_step < 5000:
            return False
        if len(recent_mel_losses) < 10:
            return False
        avg_mel_loss = sum(recent_mel_losses[-20:]) / min(20, len(recent_mel_losses))
        return avg_mel_loss < self.phase1_mel_target

    def sync_phase2_decision(self, should_transition, device=None):
        """把 rank0 的 Phase2 切换决定广播给所有训练进程。"""
        if device is None:
            device = "cpu"
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    get_backend = getattr(torch.distributed, "get_backend", None)
                    if get_backend is not None and get_backend() == "nccl" and torch.cuda.is_available():
                        device = torch.device("cuda", torch.cuda.current_device())
                except RuntimeError:
                    device = "cpu"

        decision = torch.tensor([1 if should_transition else 0], dtype=torch.int32, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(decision, src=0)
        return bool(decision.item())

    def mark_phase2(self):
        """标记进入 Phase2"""
        with open(self.phase2_marker, "w") as f:
            f.write("1")
        self.in_phase2 = True
