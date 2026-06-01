# -*- coding: utf-8 -*-
"""配置/工程/checkpoint 管理"""
import os
import re
import glob
import json
import yaml
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.join(ROOT, "projects")
PRESET_DIR = os.path.join(ROOT, "infer_presets")
SETTINGS_FILE = os.path.join(ROOT, "gui_settings.json")

for d in (PROJECT_DIR, PRESET_DIR):
    os.makedirs(d, exist_ok=True)

SPEECH_ENCODERS = ["vec768l12", "vec256l9", "hubertsoft", "whisper-ppg",
                   "cnhubertlarge", "dphubert", "whisper-ppg-large", "wavlmbase+", "wavlmlarge",
                   "etawavlmlarge"]
F0_METHODS = ["rmvpe", "fcpe", "crepe", "pm", "dio", "harvest"]
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")

# config.json 可编辑字段：(json路径, 显示名, 类型, 默认值, 范围/选项, tooltip)
CONFIG_FIELDS = [
    ("train.batch_size", "batch_size", "int", 8, (1, 64),
     "每步训练的样本数。显存大用32-48，小用4-8。"),
    ("train.learning_rate", "学习率", "float", 0.0001, (0.00001, 0.001),
     "学习率。AdamW 默认 1e-4。\n"
     "Lion 须调小到 1/3~1/10(如 2e-5~3e-5)，否则 loss 易在高位震荡不降。"),
    ("train.epochs", "epochs", "int", 10000, (100, 50000),
     "训练轮数上限。实际按step保存，此值影响学习率衰减。"),
    ("train.eval_interval", "验证/保存间隔(step)", "int", 2000, (100, 10000),
     "每N步验证一次并保存checkpoint。建议1000-3000。"),
    ("train.log_interval", "日志间隔(step)", "int", 200, (10, 1000),
     "每N步打印一次loss到终端和TensorBoard。"),
    ("train.keep_ckpts", "保留最近ckpt数", "int", 3, (1, 10),
     "自动删除旧checkpoint，只保留最新N个。0=全保留。"),
    ("train.fp16_run", "fp16_run", "bool", True, None,
     "混合精度训练。4080S建议关闭改用bf16。\n注意：开启 CQT/MRD/MBD 判别器时建议关闭(用fp32)，\n否则 nnAudio 的 CQT 变换在 fp16 下精度可能下降。"),
    ("train.half_type", "half_type", "combo", "fp16", ["fp16", "bf16"],
     "半精度类型。4080S用bf16更稳定，30/40系老卡用fp16。"),
    ("train.all_in_mem", "全量载入内存", "bool", False, None,
     "数据集全部加载到内存，加速训练但吃内存。32G+可开。"),
    ("data.sampling_rate", "采样率", "int", 44100, (22050, 48000),
     "音频采样率。官方默认44100，不建议改。"),
    ("model.speech_encoder", "内容编码器", "combo", "vec768l12", SPEECH_ENCODERS,
     "内容特征提取器，决定咬字清晰度与音色泄漏程度。\n"
     "· vec768l12：默认，唱歌综合最优，ssl_dim=768\n"
     "· wavlmlarge：WavLM-Large(取第6层)，解耦更强、咬字更准，ssl_dim需改为1024\n"
     "· etawavlmlarge：在 wavlmlarge 上做 Eta-WavLM 线性去说话人(ssl_dim=1024)，\n"
     "  音色泄漏更低。需先 `python eta_wavlm_fit.py --in_dir <多说话人wav目录>` 拟合投影\n"
     "· cnhubertlarge/whisper-ppg：偏说话\n"
     "切换后必须重新预处理(重抽特征)并重训。"),
    # ===== 判别器增强（BigVGAN-v2，仅影响训练，不改推理）=====
    ("model.use_cqt_disc", "CQT 判别器", "bool", False, None,
     "MS-SB-CQT 多尺度子带常Q变换判别器(BigVGAN-v2)。\n"
     "在对数频率轴上判别，对谐波结构/音准敏感，翻唱画质增益主要来源。\n"
     "新增依赖 nnAudio。建议翻唱优先开启此项。\n"
     "仅训练期生效，推理无任何改动。需重新训练。"),
    ("model.use_mrd_disc", "MRD 判别器", "bool", False, None,
     "多分辨率 STFT 判别器(BigVGAN-v2)。\n"
     "在多个 n_fft/hop 分辨率下判别频谱细节，补充高频质感。\n"
     "无新增依赖(torch.stft)。可与 CQT 同时开启。\n"
     "仅训练期生效，需重新训练。"),
    ("model.use_mbd_disc", "MBD 判别器", "bool", False, None,
     "多子带多尺度 STFT 判别器(BigVGAN-v2)。\n"
     "将频谱切成多个子带分别判别，进一步细化频谱。\n"
     "训练开销较大，显存紧张时可不开。\n"
     "仅训练期生效，需重新训练。"),
    # ===== 数据增强 =====
    ("train.vol_aug", "音量增强", "bool", False, None,
     "训练时随机调整音量并重算频谱，增强对响度变化的鲁棒性。\n仅作用于训练集。"),
    ("train.feature_aug", "特征域增强", "bool", False, None,
     "对已抽取的内容特征做在线扰动(噪声/时间掩码/通道dropout)。\n"
     "目的：削弱内容特征里残留的音色信息，提升音色解耦与鲁棒性。\n"
     "在 dataloader 中实时进行，改参数即时生效、无需重新预处理。\n"
     "仅作用于训练集；需配合下方三个强度参数。"),
    ("train.feature_aug_noise", "  └ 高斯噪声强度", "float", 0.0, (0.0, 0.5),
     "向内容特征加高斯噪声，强度按特征自身标准差缩放。\n"
     "0=关闭。建议 0.05~0.15，过大伤咬字。\n"
     "需先勾选「特征域增强」。"),
    ("train.feature_aug_time_mask", "  └ 时间掩码宽度(帧)", "int", 0, (0, 30),
     "随机将一段连续时间帧置零(SpecAugment 风格)，强迫模型利用上下文。\n"
     "0=关闭。表示最大掩码宽度，实际宽度在 1~该值 间随机。\n"
     "需先勾选「特征域增强」。"),
    ("train.feature_aug_channel_dropout", "  └ 通道dropout概率", "float", 0.0, (0.0, 0.5),
     "随机将整条特征通道置零并按 1/(1-p) 重缩放(保持期望幅度)。\n"
     "0=关闭。建议 0.05~0.2。\n"
     "需先勾选「特征域增强」。"),
    # ===== 高级模型结构（不常用，改动需重新训练）=====
    ("model.use_transformer_flow", "Transformer Flow", "bool", False, None,
     "用 Transformer 耦合块替换默认的残差耦合 flow，建模能力更强、显存开销更高。\n"
     "开启后受下方「Flow 参数共享」「Transformer Flow 层数」影响。\n"
     "改动模型结构，需从头重新训练。"),
    ("model.flow_share_parameter", "Flow 参数共享", "bool", False, None,
     "flow 各耦合层共享同一组参数，显著减少参数量。\n"
     "对常规与 Transformer flow 均生效。改动模型结构，需重新训练。"),
    ("model.n_layers_trans_flow", "Transformer Flow 层数", "int", 3, (1, 8),
     "Transformer flow 的层数，仅在开启「Transformer Flow」时生效。\n"
     "越大表达力越强、越慢。改动模型结构，需重新训练。"),
    ("model.use_depthwise_conv", "深度可分离卷积", "bool", False, None,
     "生成器卷积改用深度可分离卷积，参数更少、推理更快，画质略降。\n"
     "改动模型结构，需重新训练。"),
    ("model.use_automatic_f0_prediction", "自动 f0 预测", "bool", True, None,
     "内置自动音高预测分支(变声时可不输入参考音高)。\n"
     "关闭可省一点显存，但推理时需自行提供 f0。改动结构，需重新训练。"),
    ("model.speaker_embedding", "额外说话人嵌入", "bool", False, None,
     "为内容编码额外注入说话人嵌入，多说话人场景增强区分度。\n"
     "改动模型结构，需重新训练。"),
    # ===== 高级训练超参（不常用）=====
    ("train.optimizer", "优化器", "combo", "adamw", ["adamw", "lion"],
     "· adamw：默认，稳定通用。\n"
     "· lion：EvoLved Sign Momentum，更省显存、收敛快；\n"
     "  学习率取 AdamW 的 1/3~1/10，并配合更大的 weight_decay。\n"
     "切换优化器后建议重置优化器状态(从底模/G_0 重开)。"),
    ("train.weight_decay", "weight_decay", "float", 0.01, (0.0, 0.5),
     "权重衰减(L2正则)。AdamW 默认 0.01；\n"
     "Lion 通常用更大值(如 0.1~0.5)。0=关闭。"),
    ("train.warmup_epochs", "warmup epochs", "int", 0, (0, 50),
     "学习率预热轮数，按 step 换算(warmup_epochs × 每epoch步数)线性升到设定学习率。\n0=不预热。"),
    ("train.cosine_total_steps", "余弦退火总步数", "int", 40000, (1000, 500000),
     "学习率调度：warmup 之后余弦衰减，到该 step 降到底(learning_rate × 下方比例)，之后保持。\n"
     "设为你计划训练的总 step 数。训得更久就调大，否则 lr 会提前触底、几乎不更新。"),
    ("train.cosine_eta_min_ratio", "余弦最小lr比例", "float", 0.02, (0.0, 1.0),
     "余弦退火终点的学习率 = learning_rate × 此比例。\n默认 0.02(即降到 2%)。"),
    ("train.c_mel", "梅尔损失权重(c_mel)", "int", 45, (1, 100),
     "重建梅尔频谱损失权重，默认 45。调大更重视音质细节。"),
    ("train.c_kl", "KL损失权重(c_kl)", "float", 1.0, (0.1, 5.0),
     "KL 散度损失权重，默认 1.0。一般不改。"),
    ("train.seed", "随机种子", "int", 1234, (0, 999999),
     "训练随机种子，固定可复现。"),
]

# 训练参数面板分组渲染：(分组名, [json路径...], 默认展开)。仅控制 UI 布局，
# 保存/加载仍以扁平的 CONFIG_FIELDS 为准。
CONFIG_GROUPS = [
    ("基础训练", ["train.batch_size", "train.learning_rate", "train.epochs",
                  "train.eval_interval", "train.log_interval", "train.keep_ckpts",
                  "train.fp16_run", "train.half_type", "train.all_in_mem"], True),
    ("模型与编码器", ["data.sampling_rate", "model.speech_encoder"], True),
    ("判别器增强 (BigVGAN-v2，仅训练期)", ["model.use_cqt_disc", "model.use_mrd_disc",
                                          "model.use_mbd_disc"], True),
    ("数据增强", ["train.vol_aug", "train.feature_aug", "train.feature_aug_noise",
                  "train.feature_aug_time_mask", "train.feature_aug_channel_dropout"], True),
    ("高级模型结构 (不常用，改后需重训)", ["model.use_transformer_flow", "model.flow_share_parameter",
                                          "model.n_layers_trans_flow", "model.use_depthwise_conv",
                                          "model.use_automatic_f0_prediction", "model.speaker_embedding"], False),
    ("高级训练超参 (不常用)", ["train.optimizer", "train.weight_decay", "train.warmup_epochs",
                              "train.cosine_total_steps", "train.cosine_eta_min_ratio",
                              "train.c_mel", "train.c_kl", "train.seed"], False),
]

# 路径 -> 字段定义，供分组渲染查表
CONFIG_FIELD_MAP = {f[0]: f for f in CONFIG_FIELDS}

# speech_encoder -> 期望 ssl_dim（与 preprocess_flist_config.py 的设定保持一致）
ENCODER_DIM = {
    "vec768l12": 768, "dphubert": 768, "wavlmbase+": 768,
    "vec256l9": 256, "hubertsoft": 256,
    "whisper-ppg": 1024, "cnhubertlarge": 1024, "wavlmlarge": 1024, "etawavlmlarge": 1024,
    "whisper-ppg-large": 1280,
}


def check_config(cfg):
    """对 config.json 做静态体检，返回 [(level, msg), ...]，level ∈ error/warn/info/ok。
    纯规则检查、不跑训练；依据见 EXPERIMENT_UPGRADES.md 与各编码器维度约定。"""
    out = []
    t = cfg.get("train", {})
    m = cfg.get("model", {})

    # 编码器 ↔ ssl_dim（改了编码器没重新预处理的高频坑）
    enc = m.get("speech_encoder")
    exp = ENCODER_DIM.get(enc)
    if exp is not None and m.get("ssl_dim") != exp:
        out.append(("error",
            f"speech_encoder={enc} 要求 ssl_dim={exp}，当前为 {m.get('ssl_dim')}。"
            "通常是改了编码器但没重新预处理——需重抽特征并重训，否则维度不匹配会直接报错。"))

    # Lion / AdamW 超参
    opt = (t.get("optimizer") or "adamw").lower()
    lr, wd = t.get("learning_rate"), t.get("weight_decay")
    if opt == "lion":
        if lr is not None and lr > 5e-5:
            out.append(("warn", f"Lion 学习率 {lr:g} 偏大，建议 2e-5~3e-5(AdamW 的 1/3~1/10)，否则 loss 易高位震荡不降。"))
        if wd is not None and wd < 0.05:
            out.append(("warn", f"Lion 的 weight_decay={wd:g} 偏小，通常用 0.1~0.5。"))
    elif wd is not None and wd > 0.05:
        out.append(("warn", f"AdamW 的 weight_decay={wd:g} 偏大，常用 0.01 左右。"))

    # 半精度生效条件
    if t.get("half_type") == "bf16" and not t.get("fp16_run"):
        out.append(("warn", "half_type=bf16 但 fp16_run=false：半精度不生效(实际跑 fp32)。要用 bf16 须 fp16_run=true。"))

    # 增强判别器 + 半精度
    if (m.get("use_cqt_disc") or m.get("use_mrd_disc") or m.get("use_mbd_disc")) and t.get("fp16_run"):
        out.append(("info", "开了 CQT/MRD/MBD 判别器且 fp16_run=true：CQT 在半精度下精度可能下降，显存够可改 fp16_run=false 走 fp32。"))

    # 音量增强 ↔ vol_embedding
    if t.get("vol_aug") and not m.get("vol_embedding"):
        out.append(("warn", "vol_aug=true 但 model.vol_embedding=false：音量增强无法注入模型，应同时开 vol_embedding(改后需重训)。"))

    # feature_aug 开关与强度是否自洽
    augs = (t.get("feature_aug_noise", 0), t.get("feature_aug_time_mask", 0), t.get("feature_aug_channel_dropout", 0))
    if t.get("feature_aug") and not any(augs):
        out.append(("warn", "feature_aug=true 但 噪声/时间掩码/通道dropout 三个强度全为 0，增强无任何效果。"))
    elif not t.get("feature_aug") and any(augs):
        out.append(("info", "设置了 feature_aug 强度但 feature_aug=false，当前未启用。"))

    # 余弦退火总步数
    cts = t.get("cosine_total_steps", 40000)
    if cts < 5000:
        out.append(("warn", f"cosine_total_steps={cts} 偏小，lr 会很快退火触底；确认这是计划的总训练步数。"))

    # warmup
    if t.get("warmup_epochs", 0) == 0:
        out.append(("info", "warmup_epochs=0(无预热)。Lion/从零冷启建议设 3~5，稳定开局、避免初期崩。"))

    # checkpoint 占盘
    kc = t.get("keep_ckpts", 0)
    if kc > 50:
        out.append(("info", f"keep_ckpts={kc}：每对 G+D 约 1GB，{kc} 对在 {kc}GB 量级，留意磁盘。"))

    if not out:
        out.append(("ok", "未发现明显配置问题。"))
    return out


def list_projects():
    if not os.path.isdir(PROJECT_DIR):
        return []
    return sorted(d for d in os.listdir(PROJECT_DIR)
                  if os.path.isdir(os.path.join(PROJECT_DIR, d)))


def proj_dir(name):
    return os.path.join(PROJECT_DIR, name)


def config_path(name):
    return os.path.join(proj_dir(name), "config.json")


def diff_config_path(name):
    return os.path.join(proj_dir(name), "diffusion.yaml")


def filelist_dir(name):
    return os.path.join(proj_dir(name), "filelists")


def log_dir(name):
    return os.path.join(ROOT, "logs", name)


def diff_log_dir(name):
    return os.path.join(log_dir(name), "diffusion")


def create_project(name):
    name = re.sub(r"[^\w\-]+", "_", name.strip())
    if not name:
        raise ValueError("工程名为空")
    os.makedirs(os.path.join(proj_dir(name), "filelists"), exist_ok=True)
    os.makedirs(log_dir(name), exist_ok=True)
    return name


def load_config(name):
    with open(config_path(name), encoding="utf-8") as f:
        return json.load(f)


def save_config(name, values):
    """values: {json_path: new_value}"""
    cfg = load_config(name)
    changed = []
    for path, val in values.items():
        keys = path.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        old = node.get(keys[-1])
        if old != val:
            node[keys[-1]] = val
            changed.append(f"{path} = {val}")
    if changed:
        with open(config_path(name), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    return changed


def list_speakers(name):
    if not os.path.exists(config_path(name)):
        return []
    return list(load_config(name).get("spk", {}).keys())


def _step_of(p, prefix):
    m = re.search(rf"{prefix}_(\d+)\.", os.path.basename(p))
    return int(m.group(1)) if m else -1


def list_ckpts(name):
    """主模型 G_*.pth，按 step 倒序。"""
    pts = glob.glob(os.path.join(log_dir(name), "G_*.pth"))
    return sorted(pts, key=lambda p: _step_of(p, "G"), reverse=True)


def list_diff_ckpts(name):
    """扩散模型 model_*.pt，按 step 倒序。"""
    pts = glob.glob(os.path.join(diff_log_dir(name), "model_*.pt"))
    return sorted(pts, key=lambda p: _step_of(p, "model"), reverse=True)


def ckpt_label(p):
    return os.path.basename(p)


def latest_step(name):
    pts = list_ckpts(name)
    return _step_of(pts[0], "G") if pts else 0


def patch_configs_for_project(name):
    """生成配置后，把工程专属路径写进 config.json / diffusion.yaml。"""
    train_list = os.path.relpath(os.path.join(filelist_dir(name), "train.txt"), ROOT).replace("\\", "/")
    val_list = os.path.relpath(os.path.join(filelist_dir(name), "val.txt"), ROOT).replace("\\", "/")
    if os.path.exists(config_path(name)):
        cfg = load_config(name)
        cfg["data"]["training_files"] = train_list
        cfg["data"]["validation_files"] = val_list
        with open(config_path(name), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    if os.path.exists(diff_config_path(name)):
        with open(diff_config_path(name), encoding="utf-8") as f:
            dcfg = yaml.safe_load(f)
        dcfg["data"]["training_files"] = train_list
        dcfg["data"]["validation_files"] = val_list
        dcfg["env"]["expdir"] = os.path.relpath(diff_log_dir(name), ROOT).replace("\\", "/")
        with open(diff_config_path(name), "w", encoding="utf-8") as f:
            yaml.safe_dump(dcfg, f, allow_unicode=True, sort_keys=False)


def scan_dataset(root_dir):
    """扫描 {root}/{说话人}/*.<audio>，返回 [(说话人, 文件数, 时长秒), ...], 总文件数, 总时长秒。"""
    base = root_dir if os.path.isabs(root_dir) else os.path.join(ROOT, root_dir)
    if not os.path.isdir(base):
        return [], 0, 0.0
    rows, total_files, total_dur = [], 0, 0.0
    for spk in sorted(os.listdir(base)):
        spk_dir = os.path.join(base, spk)
        if not os.path.isdir(spk_dir):
            continue
        files = [f for f in os.listdir(spk_dir)
                 if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
        dur = 0.0
        for f in files:
            try:
                dur += sf.info(os.path.join(spk_dir, f)).duration
            except Exception:
                pass
        rows.append((spk, len(files), dur))
        total_files += len(files)
        total_dur += dur
    return rows, total_files, total_dur


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def list_presets():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(PRESET_DIR, "*.json")))


def save_preset(name, data):
    name = re.sub(r"[^\w\-]+", "_", name.strip())
    if not name:
        raise ValueError("预设名为空")
    with open(os.path.join(PRESET_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return name


def load_preset(name):
    with open(os.path.join(PRESET_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def delete_preset(name):
    p = os.path.join(PRESET_DIR, name + ".json")
    if os.path.exists(p):
        os.remove(p)


def read_scalars(logdir):
    """读取 TensorBoard tfevents，返回 {tag: ([steps], [values])}。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if not os.path.isdir(logdir):
        return {}
    out = {}
    for sub in (logdir, os.path.join(logdir, "eval")):
        if not os.path.isdir(sub):
            continue
        ea = EventAccumulator(sub, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            ev = ea.Scalars(tag)
            out[tag] = ([e.step for e in ev], [e.value for e in ev])
    return out
