# -*- coding: utf-8 -*-
"""声码器分类常量 + 训练配置归一化。

torch-free，供训练侧(utils/train)与 GUI(local_gui_dpg)共用，
避免 BIGVGAN_VOCODERS / 迁移逻辑在多处各抄一份。
"""

BIGVGAN_VOCODERS = ("bigvgan", "bigvgan-v2", "nsf-bigvgan-v2")
NSF_HIFIGAN_VOCODERS = ("nsf-hifigan", "nsf-snake-hifigan")

BIGVGAN_STRATEGY_DEFAULT = {
    "mode": "frozen",
    "phase1_disc_start": 90000,
    "phase1_mel_target": 0.35,
    "phase2_vocoder_lr": 1e-5,
    "use_mel_loss": True,
    "use_bigvgan_mel_loss": True,
}


def normalize_training_config(cfg, fill_defaults=False):
    """把旧配置迁移到当前 schema：
    - grad_clip_norm 从旧的 bigvgan_strategy 块上移到 train 顶层
    - 非 BigVGAN 声码器丢弃 bigvgan_strategy 块

    fill_defaults=True(GUI 编辑场景)额外补齐 train.grad_clip_norm 与
    BigVGAN 策略默认值，让面板有可编辑的初值；训练侧不填，缺失项交给代码默认。
    """
    if fill_defaults:
        train = cfg.setdefault("train", {})
        model = cfg.setdefault("model", {})
    else:
        train, model = cfg.get("train"), cfg.get("model")
        if not isinstance(train, dict) or not isinstance(model, dict):
            return cfg

    strategy = train.get("bigvgan_strategy")
    if isinstance(strategy, dict):
        if "grad_clip_norm" in strategy and "grad_clip_norm" not in train:
            train["grad_clip_norm"] = strategy["grad_clip_norm"]
        strategy.pop("grad_clip_norm", None)

    if model.get("vocoder_name") in BIGVGAN_VOCODERS:
        if fill_defaults:
            strategy = train.setdefault("bigvgan_strategy", {})
            for key, value in BIGVGAN_STRATEGY_DEFAULT.items():
                strategy.setdefault(key, value)
    else:
        train.pop("bigvgan_strategy", None)

    if fill_defaults:
        train.setdefault("grad_clip_norm", 5.0)
    return cfg
