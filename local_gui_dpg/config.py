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
                   "cnhubertlarge", "dphubert", "whisper-ppg-large", "wavlmbase+"]
F0_METHODS = ["rmvpe", "fcpe", "crepe", "pm", "dio", "harvest"]
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")

# config.json 可编辑字段：(json路径, 显示名, 类型, 默认值, 范围/选项, tooltip)
CONFIG_FIELDS = [
    ("train.batch_size", "batch_size", "int", 8, (1, 64),
     "每步训练的样本数。显存大用32-48，小用4-8。"),
    ("train.learning_rate", "学习率", "float", 0.0001, (0.00001, 0.001),
     "Adam优化器学习率。默认1e-4，过大易崩溃，过小收敛慢。"),
    ("train.epochs", "epochs", "int", 10000, (100, 50000),
     "训练轮数上限。实际按step保存，此值影响学习率衰减。"),
    ("train.eval_interval", "验证/保存间隔(step)", "int", 2000, (100, 10000),
     "每N步验证一次并保存checkpoint。建议1000-3000。"),
    ("train.log_interval", "日志间隔(step)", "int", 200, (10, 1000),
     "每N步打印一次loss到终端和TensorBoard。"),
    ("train.keep_ckpts", "保留最近ckpt数", "int", 3, (1, 10),
     "自动删除旧checkpoint，只保留最新N个。0=全保留。"),
    ("train.fp16_run", "fp16_run", "bool", True, None,
     "混合精度训练。4080S建议关闭改用bf16。"),
    ("train.half_type", "half_type", "combo", "fp16", ["fp16", "bf16"],
     "半精度类型。4080S用bf16更稳定，30/40系老卡用fp16。"),
    ("train.all_in_mem", "全量载入内存", "bool", False, None,
     "数据集全部加载到内存，加速训练但吃内存。32G+可开。"),
    ("train.vol_aug", "音量增强", "bool", False, None,
     "训练时随机调整音量，增强鲁棒性。"),
    ("data.sampling_rate", "采样率", "int", 44100, (22050, 48000),
     "音频采样率。官方默认44100，不建议改。"),
    ("model.speech_encoder", "内容编码器", "combo", "vec768l12", SPEECH_ENCODERS,
     "唱歌推荐vec768l12，说话可用hubertsoft/cnhubertlarge。"),
]


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
