# -*- coding: utf-8 -*-
"""so-vits-svc 4.1 官方版本地一体化 GUI：数据集预处理 / 训练(主+浅扩散) / 推理 / 配置 / 训练曲线 / 播放器。

工程模型：每个模型是 projects/<名>/ 下的一份 config.json + diffusion.yaml，
主模型输出到 logs/<名>/，扩散模型输出到 logs/<名>/diffusion/。
数据集预处理产物统一在官方约定的 dataset/44k（切换工程需重新预处理）。
"""
import os
import re
import glob
import json
import time
import queue
import codecs
import threading
import subprocess
import webbrowser
import sounddevice as sd
import soundfile as sf
import yaml
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import PySimpleGUI as sg

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, "venv", "python.exe")
PROJECT_DIR = os.path.join(ROOT, "projects")
DATASET_RAW = os.path.join(ROOT, "dataset_raw")
DATASET_44K = os.path.join(ROOT, "dataset", "44k")
OUT_DIR = os.path.join(ROOT, "infer_out")
PRESET_DIR = os.path.join(ROOT, "infer_presets")
SETTINGS_FILE = os.path.join(ROOT, "gui_settings.json")
for d in (PROJECT_DIR, OUT_DIR, PRESET_DIR):
    os.makedirs(d, exist_ok=True)

# 当前工程名（projects/<名>/）
PROJ = {"name": None}

SPEECH_ENCODERS = ["vec768l12", "vec256l9", "hubertsoft", "whisper-ppg",
                   "cnhubertlarge", "dphubert", "whisper-ppg-large", "wavlmbase+"]
F0_METHODS = ["rmvpe", "fcpe", "crepe", "pm", "dio", "harvest"]
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")


# ---------------- 工程管理 ----------------
def list_projects():
    if not os.path.isdir(PROJECT_DIR):
        return []
    return sorted(d for d in os.listdir(PROJECT_DIR)
                  if os.path.isdir(os.path.join(PROJECT_DIR, d)))


def proj_dir(name=None):
    return os.path.join(PROJECT_DIR, name or PROJ["name"])


def config_path(name=None):
    return os.path.join(proj_dir(name), "config.json")


def diff_config_path(name=None):
    return os.path.join(proj_dir(name), "diffusion.yaml")


def filelist_dir(name=None):
    return os.path.join(proj_dir(name), "filelists")


def log_dir(name=None):
    """主模型/tfevents 目录：logs/<工程名>。"""
    return os.path.join(ROOT, "logs", name or PROJ["name"])


def diff_log_dir(name=None):
    return os.path.join(log_dir(name), "diffusion")


def has_project():
    return bool(PROJ["name"]) and os.path.isdir(proj_dir())


def create_project(name):
    name = re.sub(r"[^\w\-]+", "_", name.strip())
    if not name:
        raise ValueError("工程名为空")
    os.makedirs(os.path.join(proj_dir(name), "filelists"), exist_ok=True)
    os.makedirs(log_dir(name), exist_ok=True)
    return name


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def _fmt_dur(sec):
    s = int(sec)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def scan_dataset(root_dir):
    """扫描 {root}/{说话人}/*.<audio>，返回 (表格行, 总文件数, 总时长秒)。"""
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
        rows.append([spk, len(files), _fmt_dur(dur)])
        total_files += len(files)
        total_dur += dur
    return rows, total_files, total_dur


# 官方 config.json 可编辑字段：(json 路径, 显示名, 类型, combo 选项)
CONFIG_FIELDS = [
    ("train.batch_size", "batch_size", "int", None),
    ("train.learning_rate", "学习率", "float", None),
    ("train.optimizer", "优化器(lion需调小lr)", "combo", ["adamw", "lion"]),
    ("train.weight_decay", "weight_decay(lion建议0.1)", "float", None),
    ("train.epochs", "epochs", "int", None),
    ("train.eval_interval", "验证/保存间隔(step)", "int", None),
    ("train.log_interval", "日志间隔(step)", "int", None),
    ("train.keep_ckpts", "保留最近ckpt数", "int", None),
    ("train.fp16_run", "fp16_run", "bool", None),
    ("train.half_type", "half_type", "combo", ["fp16", "bf16"]),
    ("train.all_in_mem", "全量载入内存(加速)", "bool", None),
    ("train.vol_aug", "音量增强(配合vol_embedding)", "bool", None),
    ("data.sampling_rate", "采样率", "int", None),
    ("model.speech_encoder", "内容编码器", "combo", SPEECH_ENCODERS),
    ("model.vol_embedding", "音量嵌入(随vol_aug)", "bool", None),
    ("model.use_transformer_flow", "Transformer Flow(吃显存/数据)", "bool", None),
    ("model.n_layers_trans_flow", "TransFlow层数(默认3)", "int", None),
    ("model.flow_share_parameter", "Flow共享参数(减容量)", "bool", None),
    ("model.use_depthwise_conv", "深度可分离卷积(减算力)", "bool", None),
    ("model.use_automatic_f0_prediction", "自动F0预测(变声用)", "bool", None),
    ("model.use_spectral_norm", "判别器谱归一化", "bool", None),
]


def _get(cfg, path):
    node = cfg
    for k in path.split("."):
        node = node[k]
    return node


def _set(cfg, path, value):
    keys = path.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def load_config():
    with open(config_path(), encoding="utf-8") as f:
        return json.load(f)


def parse_field(raw, ftype):
    if ftype == "int":
        return int(raw)
    if ftype == "float":
        return float(raw)
    if ftype == "bool":
        return str(raw).strip().lower() in ("true", "1", "yes", "on")
    return str(raw)


def save_config(vals):
    cfg = load_config()
    changed = []
    for path, _, ftype, _ in CONFIG_FIELDS:
        new = parse_field(vals[f"-CFG-{path}-"], ftype)
        try:
            old = _get(cfg, path)
        except KeyError:
            old = None
        if old != new:
            _set(cfg, path, new)
            changed.append(f"{path} = {new}")
    if changed:
        with open(config_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    return changed


def list_speakers():
    if not has_project() or not os.path.exists(config_path()):
        return []
    return list(load_config().get("spk", {}).keys())


# 推理参数预设保存的界面 key
PRESET_KEYS = ["-T-", "-SDB-", "-FM-", "-APF-", "-CR-", "-NS-", "-PAD-", "-CLIP-",
               "-DEV-", "-SPK-", "-SHD-", "-KSTEP-", "-SE-", "-EH-", "-LEA-", "-FR-"]
INFER_SETTING_KEYS = PRESET_KEYS + ["-IN-", "-CKPT-", "-CM-", "-PRESET-", "-PRESET-NAME-"]

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


class TermBuffer:
    """迷你终端：处理 \\r 覆盖当前行、\\n 换行，剥 ANSI，行数封顶。"""

    MAX_LINES = 400

    def __init__(self):
        self.lines = [""]
        self.lock = threading.Lock()
        self.dirty = True

    def feed(self, text):
        text = ANSI_RE.sub("", text).replace("\r\n", "\n")
        with self.lock:
            for ch in text:
                if ch == "\n":
                    self.lines.append("")
                elif ch == "\r":
                    self.lines[-1] = ""
                else:
                    self.lines[-1] += ch
            if len(self.lines) > self.MAX_LINES:
                self.lines = self.lines[-self.MAX_LINES:]
            self.dirty = True

    def snapshot(self):
        with self.lock:
            self.dirty = False
            return "\n".join(self.lines)

    def clear(self):
        with self.lock:
            self.lines = [""]
            self.dirty = True


class Job:
    """单个后台子进程：stdout -> TermBuffer，结束 -> done_q。"""

    def __init__(self):
        self.proc = None
        self.buf = TermBuffer()
        self.done_q = queue.Queue()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd, header=""):
        if self.running():
            self.buf.feed("[忽略] 已有进程在运行，请先停止。\n")
            return
        self.buf.feed((header or "") + "$ " + " ".join(cmd) + "\n\n")
        threading.Thread(target=self._run, args=(cmd,), daemon=True).start()

    def start_chain(self, steps):
        """串行执行 [(标题, cmd), ...]，任一步非 0 退出即中止后续。"""
        if self.running():
            self.buf.feed("[忽略] 已有进程在运行，请先停止。\n")
            return
        threading.Thread(target=self._run_chain, args=(steps,), daemon=True).start()

    def _run_chain(self, steps):
        for i, (title, cmd) in enumerate(steps, 1):
            self.buf.feed(f"\n===== [{i}/{len(steps)}] {title} =====\n$ "
                          + " ".join(cmd) + "\n\n")
            rc = self._run(cmd, signal_done=False)
            if rc != 0:
                self.buf.feed(f"\n[中止] 步骤「{title}」失败(退出码 {rc})，停止后续。\n")
                self.done_q.put(rc)
                return
        self.buf.feed("\n===== 全流程完成 =====\n")
        self.done_q.put(0)

    def _run(self, cmd, signal_done=True):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   PYTHONPATH=ROOT)
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, env=env)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                chunk = self.proc.stdout.read(1024)
                if not chunk:
                    break
                self.buf.feed(decoder.decode(chunk))
            self.buf.feed(decoder.decode(b"", final=True))
            self.proc.wait()
            rc = self.proc.returncode
            self.buf.feed(f"\n--- 退出码 {rc} ---\n")
            if signal_done:
                self.done_q.put(rc)
            return rc
        except Exception as e:
            self.buf.feed(f"\n[异常] {e}\n")
            if signal_done:
                self.done_q.put(-1)
            return -1

    def stop(self):
        if self.running():
            self.proc.terminate()
            return "已发送停止信号。\n"
        return "当前没有正在运行的进程。\n"

    def drain_to(self, window, key):
        if self.buf.dirty:
            window[key].update(self.buf.snapshot())


# ---------------- checkpoint ----------------
def _step_of(p, prefix):
    m = re.search(rf"{prefix}_(\d+)\.", os.path.basename(p))
    return int(m.group(1)) if m else -1


def list_ckpts():
    """主模型 G_*.pth，按 step 倒序。"""
    if not has_project():
        return []
    pts = glob.glob(os.path.join(log_dir(), "G_*.pth"))
    return sorted(pts, key=lambda p: _step_of(p, "G"), reverse=True)


def list_diff_ckpts():
    """扩散模型 model_*.pt，按 step 倒序。"""
    if not has_project():
        return []
    pts = glob.glob(os.path.join(diff_log_dir(), "model_*.pt"))
    return sorted(pts, key=lambda p: _step_of(p, "model"), reverse=True)


def ckpt_label(p):
    return os.path.basename(p)


def latest_step():
    pts = list_ckpts()
    return _step_of(pts[0], "G") if pts else 0


# ---------------- 播放器 ----------------
class Player:
    """sounddevice 播放器：播放/暂停/停止/跳转/音量。"""

    def __init__(self):
        self.data = None
        self.sr = 0
        self.pos = 0
        self.stream = None
        self.volume = 1.0
        self.lock = threading.Lock()

    def load(self, path):
        self.stop()
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        with self.lock:
            self.data, self.sr, self.pos = data, sr, 0

    @property
    def total(self):
        return 0 if self.data is None else len(self.data)

    @property
    def duration(self):
        return self.total / self.sr if self.sr else 0

    @property
    def cur_time(self):
        return self.pos / self.sr if self.sr else 0

    def playing(self):
        return self.stream is not None and self.stream.active

    def _callback(self, outdata, frames, time_info, status):
        with self.lock:
            end = self.pos + frames
            chunk = self.data[self.pos:end]
            self.pos = min(end, self.total)
        n = len(chunk)
        outdata[:n] = chunk * self.volume
        if n < frames:
            outdata[n:] = 0
            raise sd.CallbackStop

    def play(self):
        if self.data is None or self.playing():
            return
        if self.pos >= self.total:
            self.pos = 0
        self.stream = sd.OutputStream(
            samplerate=self.sr, channels=self.data.shape[1],
            callback=self._callback, finished_callback=self._on_finish)
        self.stream.start()

    def _on_finish(self):
        if self.pos >= self.total:
            self.pos = 0

    def pause(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def stop(self):
        self.pause()
        with self.lock:
            self.pos = 0

    def seek(self, frac):
        with self.lock:
            self.pos = int(max(0.0, min(1.0, frac)) * self.total)

    def set_volume(self, vol_pct):
        self.volume = max(0.0, vol_pct / 100.0)


# ---------------- 训练曲线 ----------------
def read_scalars(logdir):
    """官方 SummaryWriter 直接写在 logs/<名>/ 与 logs/<名>/eval/。合并读取。"""
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


class Chart:
    """matplotlib 曲线嵌入 PySimpleGUI Canvas。"""

    def __init__(self):
        self.fig = Figure(figsize=(8.8, 3.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.agg = None
        self.scalars = {}

    def attach(self, canvas_elem):
        self.agg = FigureCanvasTkAgg(self.fig, canvas_elem.Widget)
        self.agg.draw()
        self.agg.get_tk_widget().pack(side="top", fill="both", expand=1)

    def reload(self, logdir):
        self.scalars = read_scalars(logdir)
        return list(self.scalars)

    def plot(self, tag, logy=False):
        self.ax.clear()
        last = ""
        if tag in self.scalars:
            steps, vals = self.scalars[tag]
            self.ax.plot(steps, vals, lw=1.0, color="#4FC3F7")
            self.ax.set_title(tag)
            self.ax.set_xlabel("step")
            if logy:
                self.ax.set_yscale("log")
            self.ax.grid(True, alpha=0.3)
            if steps:
                last = f"step {steps[-1]}  =  {vals[-1]:.5f}"
        self.fig.tight_layout()
        if self.agg:
            self.agg.draw()
        return last


# ---------------- 推理参数预设 ----------------
def list_presets():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(PRESET_DIR, "*.json")))


def save_preset(name, vals):
    name = re.sub(r"[^\w\-]+", "_", name.strip())
    if not name:
        raise ValueError("预设名为空")
    data = {k: vals[k] for k in PRESET_KEYS if k in vals}
    with open(os.path.join(PRESET_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return name


def load_preset(name, window):
    with open(os.path.join(PRESET_DIR, name + ".json"), encoding="utf-8") as f:
        data = json.load(f)
    for k, v in data.items():
        if k in PRESET_KEYS:
            window[k].update(v)


def infer_settings(vals):
    return {k: vals[k] for k in INFER_SETTING_KEYS if k in vals}


def save_infer_settings(vals):
    settings = load_settings()
    settings["infer"] = infer_settings(vals)
    save_settings(settings)


def infer_saved(key, default=None):
    return load_settings().get("infer", {}).get(key, default)


def apply_infer_settings(window):
    for key, value in load_settings().get("infer", {}).items():
        if key in window.AllKeysDict:
            window[key].update(value=value)


def delete_preset(name):
    p = os.path.join(PRESET_DIR, name + ".json")
    if os.path.exists(p):
        os.remove(p)


# ---------------- 命令构造（官方独立脚本） ----------------
def _py(script):
    return [PYTHON, os.path.join(ROOT, script)]


def _rel(p):
    return os.path.relpath(p, ROOT)


def resample_cmd(vals):
    return _py("resample.py") + [
        "--in_dir", _rel(DATASET_RAW),
        "--out_dir2", _rel(DATASET_44K),
        "--sr2", str(int(vals["-RS-SR-"]))]


def config_cmd(vals):
    cmd = _py("preprocess_flist_config.py") + [
        "--source_dir", _rel(DATASET_44K),
        "--speech_encoder", vals["-ENC-"],
        "--train_list", _rel(os.path.join(filelist_dir(), "train.txt")),
        "--val_list", _rel(os.path.join(filelist_dir(), "val.txt")),
        "--config_out", _rel(config_path()),
        "--diff_config_out", _rel(diff_config_path())]
    if vals.get("-VOLAUG-"):
        cmd += ["--vol_aug"]
    return cmd


def hubert_cmd(vals):
    cmd = _py("preprocess_hubert_f0.py") + [
        "--in_dir", _rel(DATASET_44K),
        "--config", _rel(config_path()),
        "--diff-config", _rel(diff_config_path()),
        "--f0_predictor", vals["-HB-FM-"]]
    if str(vals["-HB-N-"]).strip():
        cmd += ["--num_processes", str(int(vals["-HB-N-"]))]
    if vals.get("-USE-DIFF-"):
        cmd += ["--use_diff"]
    return cmd


def patch_configs_for_project():
    """生成配置后，把工程专属路径写进 config.json / diffusion.yaml。
    官方脚本按模板默认值写 diffusion 的 filelists/expdir，多工程时需修正。"""
    train_list = _rel(os.path.join(filelist_dir(), "train.txt")).replace("\\", "/")
    val_list = _rel(os.path.join(filelist_dir(), "val.txt")).replace("\\", "/")
    # config.json：训练/验证列表
    if os.path.exists(config_path()):
        cfg = load_config()
        cfg["data"]["training_files"] = train_list
        cfg["data"]["validation_files"] = val_list
        with open(config_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    # diffusion.yaml：训练/验证列表 + 输出目录
    if os.path.exists(diff_config_path()):
        with open(diff_config_path(), encoding="utf-8") as f:
            dcfg = yaml.safe_load(f)
        dcfg["data"]["training_files"] = train_list
        dcfg["data"]["validation_files"] = val_list
        dcfg["env"]["expdir"] = _rel(diff_log_dir()).replace("\\", "/")
        with open(diff_config_path(), "w", encoding="utf-8") as f:
            yaml.safe_dump(dcfg, f, allow_unicode=True, sort_keys=False)


def train_cmd():
    """主模型训练：python train.py -c <工程config> -m <工程名>。"""
    return _py("train.py") + ["-c", _rel(config_path()), "-m", PROJ["name"]]


def diff_train_cmd():
    """浅扩散训练：python train_diff.py -c <工程diffusion.yaml>。"""
    return _py("train_diff.py") + ["-c", _rel(diff_config_path())]


def updated_audio_files(out_dir, before):
    audio_exts = (".wav", ".flac")
    files = glob.glob(os.path.join(out_dir or "", "*"))
    updated = []
    for path in files:
        if not path.lower().endswith(audio_exts):
            continue
        mtime = os.path.getmtime(path)
        if path not in before or mtime > before[path]:
            updated.append(path)
    return updated


def keep_selection(values, current, saved=None):
    if current in values:
        return current
    if saved in values:
        return saved
    return next(iter(values), "")


def infer_cmd(vals, model_path, out_path):
    device = "cuda:0" if vals["-DEV-"] == "GPU" else "cpu"
    cmd = _py("inference_main.py") + [
        "-m", _rel(model_path),
        "-c", _rel(config_path()),
        "-ip", vals["-IN-"],
        "-op", os.path.dirname(out_path),
        "-t", str(int(vals["-T-"])),
        "-s", vals["-SPK-"],
        "-sd", str(int(float(vals["-SDB-"]))),
        "-ns", str(vals["-NS-"]),
        "-p", str(vals["-PAD-"]),
        "-cl", str(vals["-CLIP-"]),
        "-f0p", vals["-FM-"],
        "-d", device,
        "-wf", "wav",
        "-lea", str(vals["-LEA-"])]
    if vals["-APF-"]:
        cmd += ["-a"]
    if vals.get("-EH-"):
        cmd += ["-eh"]
    if vals.get("-FR-"):
        cmd += ["-fr"]
    # 聚类/特征检索
    if vals["-CM-"] and os.path.exists(vals["-CM-"]) and float(vals["-CR-"] or 0) > 0:
        cmd += ["-cm", _rel(vals["-CM-"]), "-cr", str(vals["-CR-"])]
    # 浅扩散
    if vals.get("-SHD-"):
        diff_pts = list_diff_ckpts()
        if diff_pts:
            cmd += ["-shd", "-dm", _rel(diff_pts[0]),
                    "-dc", _rel(diff_config_path()),
                    "-ks", str(int(vals["-KSTEP-"]))]
            if vals.get("-SE-"):
                cmd += ["-se"]
    return cmd


# ---------------- 标签页布局 ----------------
def project_bar():
    """顶部工程选择条，所有页共享。"""
    return [sg.Frame("当前工程", [[
        sg.Text("工程"),
        sg.Combo(list_projects(), key="-PROJ-", readonly=True, size=(24, 1),
                 enable_events=True, default_value=PROJ["name"] or ""),
        sg.Button("刷新", key="-PROJ-REFRESH-"),
        sg.Text("新建"),
        sg.Input(key="-PROJ-NEW-", size=(16, 1)),
        sg.Button("创建工程", key="-PROJ-CREATE-", button_color=("white", "#1565C0")),
        sg.Text("", key="-PROJ-MSG-", text_color="yellow", expand_x=True),
    ]], expand_x=True)]


def tab_dataset():
    cpu = os.cpu_count() or 4
    return [
        [sg.Frame("数据集概览", [
            [sg.Text("源目录"),
             sg.Input(_rel(DATASET_RAW), key="-DS-SCANDIR-", size=(28, 1)),
             sg.FolderBrowse("浏览", target="-DS-SCANDIR-", initial_folder=ROOT),
             sg.Button("扫描", key="-DS-SCAN-", button_color=("white", "#1565C0")),
             sg.Text("", key="-DS-SCAN-MSG-", text_color="yellow", expand_x=True)],
            [sg.Table(values=[], headings=["说话人", "文件数", "时长"],
                      key="-DS-TABLE-", num_rows=5, justification="left",
                      col_widths=[24, 8, 12], auto_size_columns=False, expand_x=True)],
            [sg.Text("训练音频放到 dataset_raw/{说话人}/*.wav，每个说话人一个子目录。",
                     text_color="gray")],
        ], expand_x=True)],
        [sg.Frame("第一步：重采样到 44.1kHz (resample.py)", [
            [sg.Text("dataset_raw → dataset/44k，自动去静音+响度归一。")],
            [sg.Text("目标采样率"), sg.Input("44100", key="-RS-SR-", size=(8, 1)),
             sg.Button("重采样", key="-RESAMPLE-", button_color=("white", "green"))],
        ], expand_x=True)],
        [sg.Frame("第二步：生成配置 (preprocess_flist_config.py)", [
            [sg.Text("内容编码器"),
             sg.Combo(SPEECH_ENCODERS, "vec768l12", key="-ENC-", readonly=True, size=(16, 1)),
             sg.Checkbox("音量增强(vol_aug)", key="-VOLAUG-"),
             sg.Button("生成配置", key="-PRECONFIG-", button_color=("white", "green"))],
            [sg.Text("唱歌推荐 vec768l12。生成 config.json + diffusion.yaml 到工程目录。",
                     text_color="gray")],
        ], expand_x=True)],
        [sg.Frame("第三步：提取特征+f0 (preprocess_hubert_f0.py)", [
            [sg.Text("f0 提取器"),
             sg.Combo(F0_METHODS, "rmvpe", key="-HB-FM-", readonly=True, size=(10, 1)),
             sg.Text("并行进程"),
             sg.Spin(list(range(0, cpu + 1)), 1, key="-HB-N-", size=(4, 1)),
             sg.Checkbox("同时提取扩散特征(--use_diff)", key="-USE-DIFF-", default=True),
             sg.Button("提取特征", key="-HUBERT-", button_color=("white", "green"))],
            [sg.Text("唱歌推荐 rmvpe。勾选扩散特征后才能训练浅扩散模型。",
                     text_color="gray")],
        ], expand_x=True)],
        [sg.Frame("一键全流程", [
            [sg.Button("一键预处理", key="-DS-ALL-", button_color=("white", "#AD1457"),
                       size=(14, 1)),
             sg.Button("停止", key="-DS-STOP-"),
             sg.Text("依次执行 重采样→生成配置→提取特征。", text_color="gray"),
             sg.Text("", key="-DS-PIPE-MSG-", text_color="yellow", expand_x=True)],
        ], expand_x=True)],
        [sg.Multiline(key="-DS-LOG-", size=(100, 10), autoscroll=True, disabled=True,
                      font=("Consolas", 9), expand_x=True, expand_y=True)],
    ]


def config_panel():
    rows, line = [], []
    for path, name, ftype, choices in CONFIG_FIELDS:
        key = f"-CFG-{path}-"
        label = sg.Text(name, size=(18, 1))
        if ftype == "bool":
            field = sg.Checkbox("", key=key,
                                enable_events=(path == "train.fp16_run"))
        elif ftype == "combo":
            field = sg.Combo(choices, key=key, size=(16, 1))
        else:
            field = sg.Input(key=key, size=(18, 1))
        col_kwargs = {"pad": (6, 2)}
        if path == "train.half_type":
            col_kwargs["key"] = "-CFGCOL-half_type-"
            col_kwargs["visible"] = False
        line.append(sg.Column([[label, field]], **col_kwargs))
        if len(line) == 2:
            rows.append(line)
            line = []
    if line:
        rows.append(line)
    return rows


def tab_train():
    return [
        [sg.Frame("主模型训练 (train.py)", [
            [sg.Checkbox("启动 TensorBoard", key="-TR-TB-", default=True),
             sg.Text("到达step自动停(0=不限)"),
             sg.Input("0", key="-TR-TARGET-", size=(8, 1)),
             sg.Button("开始训练", key="-TRAIN-", button_color=("white", "green")),
             sg.Button("停止", key="-TRAIN-STOP-"),
             sg.Text("", key="-TRAIN-MSG-", text_color="yellow")],
            [sg.Text("自动从 logs/<工程> 最新 G/D_*.pth 续训；需手动放底模 G_0.pth/D_0.pth。",
                     text_color="gray")],
        ], expand_x=True)],
        [sg.Frame("浅扩散训练 (train_diff.py，可选)", [
            [sg.Button("开始扩散训练", key="-DIFF-TRAIN-", button_color=("white", "#6A1B9A")),
             sg.Button("停止", key="-DIFF-STOP-"),
             sg.Text("", key="-DIFF-MSG-", text_color="yellow")],
            [sg.Text("需先在特征提取勾选 --use_diff。扩散模型存到 logs/<工程>/diffusion。",
                     text_color="gray")],
        ], expand_x=True)],
        [sg.Frame("训练参数 (config.json)", [
            *config_panel(),
            [sg.Button("重新加载", key="-CFG-RELOAD-"),
             sg.Button("保存到 config.json", key="-CFG-SAVE-", button_color=("white", "green")),
             sg.Text("4080S 建议 fp16_run=true + half_type=bf16(此组合才启用bf16混合精度)。",
                     text_color="#80CBC4"),
             sg.Text("", key="-CFG-MSG-", text_color="yellow", expand_x=True)],
        ], expand_x=True)],
        [sg.TabGroup([[
            sg.Tab("训练日志", [
                [sg.Multiline(key="-TRAIN-LOG-", autoscroll=True, disabled=True,
                              font=("Consolas", 9), expand_x=True, expand_y=True)],
            ]),
            sg.Tab("扩散日志", [
                [sg.Multiline(key="-DIFF-LOG-", autoscroll=True, disabled=True,
                              font=("Consolas", 9), expand_x=True, expand_y=True)],
            ]),
            sg.Tab("训练曲线", [
                [sg.Text("曲线"),
                 sg.Combo([], key="-CHART-TAG-", readonly=True, size=(28, 1), enable_events=True),
                 sg.Checkbox("对数纵轴", key="-CHART-LOG-", enable_events=True),
                 sg.Checkbox("自动刷新(5s)", key="-CHART-AUTO-", default=True),
                 sg.Button("刷新曲线", key="-CHART-REFRESH-"),
                 sg.Button("浏览器打开TB", key="-TB-OPEN-")],
                [sg.Canvas(key="-CHART-CANVAS-", expand_x=True, expand_y=True)],
                [sg.Text("", key="-CHART-MSG-", text_color="yellow", expand_x=True)],
            ]),
        ]], expand_x=True, expand_y=True)],
    ]


def tab_infer():
    labels = [ckpt_label(p) for p in list_ckpts()]
    spks = list_speakers()
    common = [
        [sg.Text("输入音频", size=(8, 1)), sg.Input(key="-IN-", expand_x=True),
         sg.FileBrowse("选择", file_types=(("音频", "*.wav *.flac *.mp3 *.ogg"),))],
        [sg.Text("模型", size=(8, 1)),
         sg.Combo(labels, key="-CKPT-", expand_x=True,
                  default_value=labels[0] if labels else ""),
         sg.Button("刷新", key="-CKPT-REFRESH-")],
        [sg.Text("说话人", size=(8, 1)),
         sg.Combo(spks, key="-SPK-", size=(18, 1),
                  default_value=spks[0] if spks else ""),
         sg.Text("设备"),
         sg.Combo(["GPU", "CPU"], "GPU", key="-DEV-", size=(6, 1), readonly=True),
         sg.Text("变调(半音)"),
         sg.Slider((-24, 24), 0, 1, orientation="h", key="-T-", size=(22, 15))],
    ]
    pitch = [
        [sg.Text("f0 提取器"), sg.Combo(F0_METHODS, "rmvpe", key="-FM-", size=(10, 1)),
         sg.Checkbox("自动预测f0(说话用,唱歌关)", key="-APF-", default=False),
         sg.Text("切片阈值(dB)"), sg.Input("-40", key="-SDB-", size=(6, 1)),
         sg.Text("噪声scale"), sg.Input("0.4", key="-NS-", size=(5, 1))],
    ]
    diffusion = [
        [sg.Checkbox("浅扩散(改善电音/音质)", key="-SHD-"),
         sg.Text("扩散步数k_step"), sg.Input("100", key="-KSTEP-", size=(6, 1)),
         sg.Checkbox("二次编码", key="-SE-"),
         sg.Checkbox("NSF-HIFIGAN增强器", key="-EH-"),
         sg.Checkbox("特征检索", key="-FR-")],
        [sg.Text("浅扩散自动用 logs/<工程>/diffusion 下最新模型；与增强器互斥。",
                 text_color="gray")],
    ]
    cluster = [
        [sg.Text("聚类/检索模型"), sg.Input(key="-CM-", size=(40, 1)),
         sg.FileBrowse("选择", file_types=(("模型", "*.pt *.pth *.pkl"),)),
         sg.Text("混合比例"), sg.Input("0", key="-CR-", size=(5, 1))],
    ]
    advanced = [
        [sg.Text("pad(秒)"), sg.Input("0.5", key="-PAD-", size=(5, 1)),
         sg.Text("强制切片(秒,0=自动)"), sg.Input("0", key="-CLIP-", size=(5, 1)),
         sg.Text("响度包络(0-1)"), sg.Input("1", key="-LEA-", size=(5, 1))],
    ]
    preset = [
        [sg.Text("预设"),
         sg.Combo(list_presets(), key="-PRESET-", size=(22, 1), readonly=True),
         sg.Button("加载", key="-PRESET-LOAD-"), sg.Button("刷新", key="-PRESET-REFRESH-"),
         sg.Text("另存为"), sg.Input(key="-PRESET-NAME-", size=(14, 1)),
         sg.Button("保存", key="-PRESET-SAVE-"), sg.Button("删除", key="-PRESET-DEL-"),
         sg.Text("", key="-PRESET-MSG-", text_color="yellow")],
    ]
    player = [
        [sg.Button("▶ 播放", key="-PLAY-", size=(8, 1)),
         sg.Button("⏸ 暂停", key="-PAUSE-", size=(8, 1)),
         sg.Button("⏹ 停止", key="-PSTOP-", size=(6, 1)),
         sg.Text("00:00", key="-PTIME-", size=(6, 1)),
         sg.Slider((0, 1000), 0, 1, orientation="h", key="-SEEK-", size=(34, 12),
                   enable_events=True, disable_number_display=True),
         sg.Text("时长", key="-PDUR-", size=(6, 1))],
        [sg.Text("音量", size=(4, 1)),
         sg.Slider((0, 150), 100, 1, orientation="h", key="-VOL-", size=(20, 12),
                   enable_events=True, disable_number_display=True),
         sg.Text("100%", key="-VOLPCT-", size=(5, 1)),
         sg.Text("", key="-OUTPATH-", text_color="yellow", expand_x=True)],
    ]
    return [
        [sg.Frame("常用", common, expand_x=True)],
        [sg.Frame("音高 / f0", pitch, expand_x=True)],
        [sg.Frame("浅扩散 / 增强 (官方)", diffusion, expand_x=True)],
        [sg.Frame("聚类/特征检索(可选)", cluster, expand_x=True)],
        [sg.Frame("高级", advanced, expand_x=True)],
        [sg.Frame("参数预设", preset, expand_x=True)],
        [sg.Button("开始转换", key="-RUN-", button_color=("white", "green"), size=(12, 1))],
        [sg.Frame("播放器", player, expand_x=True)],
        [sg.Multiline(key="-INF-LOG-", size=(100, 8), autoscroll=True, disabled=True,
                      font=("Consolas", 9), expand_x=True, expand_y=True)],
    ]


def build_window():
    sg.theme("DarkBlue3")
    settings = load_settings()
    projs = list_projects()
    saved = settings.get("project")
    PROJ["name"] = saved if saved in projs else (projs[0] if projs else None)
    layout = [
        project_bar(),
        [sg.TabGroup([[
            sg.Tab("数据集预处理", tab_dataset()),
            sg.Tab("训练", tab_train()),
            sg.Tab("推理", tab_infer()),
        ]], expand_x=True, expand_y=True)],
    ]
    window = sg.Window("so-vits-svc 4.1 一体化工具", layout, finalize=True, resizable=True)
    window["-PROJ-"].update(values=projs, value=PROJ["name"] or "")
    vol = settings.get("volume", 100)
    window["-VOL-"].update(vol)
    window["-VOLPCT-"].update(f"{int(vol)}%")
    window["-DEV-"].update(settings.get("device", "GPU"))
    window["-TR-TB-"].update(settings.get("tensorboard", True))
    window["-CHART-AUTO-"].update(settings.get("chart_auto", True))
    window["-CHART-LOG-"].update(settings.get("chart_log", False))
    if has_project() and os.path.exists(config_path()):
        refresh_config_fields(window)
    chart = Chart()
    chart.attach(window["-CHART-CANVAS-"])
    return window, chart


def refresh_config_fields(window):
    if not has_project() or not os.path.exists(config_path()):
        return
    cfg = load_config()
    for path, _, ftype, _ in CONFIG_FIELDS:
        try:
            val = _get(cfg, path)
        except KeyError:
            continue
        if ftype == "bool":
            window[f"-CFG-{path}-"].update(bool(val))
        else:
            window[f"-CFG-{path}-"].update(str(val))
    sync_half_type_visible(window)


def sync_half_type_visible(window):
    window["-CFGCOL-half_type-"].update(visible=bool(window["-CFG-train.fp16_run-"].get()))


def init_chart(window, chart):
    tags = chart.reload(log_dir()) if has_project() else []
    default = "loss/g/total" if "loss/g/total" in tags else (tags[0] if tags else "")
    window["-CHART-TAG-"].update(values=tags, value=default)
    if default:
        window["-CHART-MSG-"].update(chart.plot(default))


def refresh_project_ui(window, ckpt_map, state):
    """切换/创建工程后，刷新所有与工程相关的界面元素。"""
    current_ckpt = window["-CKPT-"].get()
    current_spk = window["-SPK-"].get()
    saved = load_settings().get("infer", {})
    ckpt_map.clear()
    ckpt_map.update({ckpt_label(p): p for p in list_ckpts()})
    window["-CKPT-"].update(values=list(ckpt_map),
                            value=keep_selection(ckpt_map, current_ckpt, saved.get("-CKPT-")))
    window["-SPK-"].update(values=list_speakers(),
                           value=keep_selection(list_speakers(), current_spk, saved.get("-SPK-")))
    refresh_config_fields(window)
    init_chart(window, state["chart"])


def _fmt_time(sec):
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def update_player_ui(window, state):
    player = state["player"]
    if player.data is None:
        return
    window["-PDUR-"].update(_fmt_time(player.duration))
    window["-PTIME-"].update(_fmt_time(player.cur_time))
    if player.playing() and player.total:
        window["-SEEK-"].update(int(player.cur_time / player.duration * 1000))


# ---------------- 事件处理 ----------------
def do_scan(vals, window):
    rows, n, dur = scan_dataset(vals["-DS-SCANDIR-"])
    window["-DS-TABLE-"].update(values=rows)
    if rows:
        window["-DS-SCAN-MSG-"].update(
            f"{len(rows)} 个说话人，共 {n} 个文件，总时长 {_fmt_dur(dur)}")
    else:
        window["-DS-SCAN-MSG-"].update("未找到说话人子目录或音频文件。", text_color="red")


def handle_project(event, vals, window, ckpt_map, state):
    if event == "-PROJ-":
        if vals["-PROJ-"]:
            PROJ["name"] = vals["-PROJ-"]
            refresh_project_ui(window, ckpt_map, state)
            window["-PROJ-MSG-"].update(f"已切换到工程 {PROJ['name']}")
    elif event == "-PROJ-REFRESH-":
        window["-PROJ-"].update(values=list_projects(), value=PROJ["name"] or "")
    elif event == "-PROJ-CREATE-":
        try:
            name = create_project(vals["-PROJ-NEW-"])
        except ValueError as e:
            window["-PROJ-MSG-"].update(str(e), text_color="red")
            return
        PROJ["name"] = name
        window["-PROJ-"].update(values=list_projects(), value=name)
        window["-PROJ-NEW-"].update("")
        refresh_project_ui(window, ckpt_map, state)
        window["-PROJ-MSG-"].update(f"已创建并切换到工程 {name}")


def _need_project(job, window, msg_key):
    if not has_project():
        window[msg_key].update("请先创建/选择工程。", text_color="red")
        return False
    return True


def handle_dataset(event, vals, jobs, window, state):
    job = jobs["ds"]
    if event == "-DS-SCAN-":
        do_scan(vals, window)
    elif event in ("-RESAMPLE-", "-PRECONFIG-", "-HUBERT-", "-DS-ALL-"):
        if not _need_project(job, window, "-DS-PIPE-MSG-"):
            return
        if event == "-RESAMPLE-":
            job.start(resample_cmd(vals))
        elif event == "-PRECONFIG-":
            state["patch_after_ds"] = True
            job.start(config_cmd(vals))
        elif event == "-HUBERT-":
            job.start(hubert_cmd(vals))
        elif event == "-DS-ALL-":
            job.buf.clear()
            state["patch_after_ds"] = True
            window["-DS-PIPE-MSG-"].update("一键全流程运行中…")
            job.start_chain([
                ("重采样", resample_cmd(vals)),
                ("生成配置", config_cmd(vals)),
                ("提取特征+f0", hubert_cmd(vals)),
            ])
    elif event == "-DS-STOP-":
        job.buf.feed(job.stop())


def handle_config(event, vals, window):
    if event == "-CFG-train.fp16_run-":
        sync_half_type_visible(window)
    elif event == "-CFG-RELOAD-":
        if has_project() and os.path.exists(config_path()):
            refresh_config_fields(window)
            window["-CFG-MSG-"].update("已从配置文件重新加载。")
        else:
            window["-CFG-MSG-"].update("配置不存在，先跑预处理。", text_color="red")
    elif event == "-CFG-SAVE-":
        if not has_project() or not os.path.exists(config_path()):
            window["-CFG-MSG-"].update("配置不存在，先跑预处理。", text_color="red")
            return
        try:
            changed = save_config(vals)
        except (ValueError, TypeError) as e:
            window["-CFG-MSG-"].update(f"保存失败：{e}", text_color="red")
            return
        window["-CFG-MSG-"].update(
            f"已保存 {len(changed)} 项改动。" if changed else "无改动。", text_color="yellow")


def handle_train(event, vals, jobs, window, state):
    if event == "-TRAIN-":
        if jobs["train"].running():
            window["-TRAIN-MSG-"].update("训练已在运行。")
            return
        if not has_project() or not os.path.exists(config_path()):
            window["-TRAIN-MSG-"].update("配置不存在，先完成预处理。", text_color="red")
            return
        target = str(vals.get("-TR-TARGET-", "")).strip()
        state["target_step"] = int(target) if target.isdigit() and int(target) > 0 else 0
        step = latest_step()
        tip = f"从 step {step} 续训..." if step else "从底模/零开始训练..."
        if state["target_step"]:
            tip += f" 到 {state['target_step']} step 自动停。"
        window["-TRAIN-MSG-"].update(tip)
        if vals["-TR-TB-"] and not jobs["tb"].running():
            jobs["tb"].start([PYTHON, "-m", "tensorboard.main",
                              "--logdir", _rel(log_dir()), "--port", "6006"])
        jobs["train"].start(train_cmd())
    elif event == "-TRAIN-STOP-":
        state["target_step"] = 0
        window["-TRAIN-MSG-"].update(jobs["train"].stop())
    elif event == "-DIFF-TRAIN-":
        if not has_project() or not os.path.exists(diff_config_path()):
            window["-DIFF-MSG-"].update("扩散配置不存在，先跑预处理(勾选--use_diff)。",
                                        text_color="red")
            return
        jobs["diff"].start(diff_train_cmd())
        window["-DIFF-MSG-"].update("扩散训练启动...")
    elif event == "-DIFF-STOP-":
        window["-DIFF-MSG-"].update(jobs["diff"].stop())
    elif event in ("-CHART-REFRESH-", "-CHART-TAG-", "-CHART-LOG-"):
        chart = state["chart"]
        if event == "-CHART-REFRESH-":
            tags = chart.reload(log_dir())
            cur = vals["-CHART-TAG-"] if vals["-CHART-TAG-"] in tags else (tags[0] if tags else "")
            window["-CHART-TAG-"].update(values=tags, value=cur)
        cur = vals["-CHART-TAG-"]
        if cur:
            window["-CHART-MSG-"].update(chart.plot(cur, vals["-CHART-LOG-"]))
    elif event == "-TB-OPEN-":
        if not jobs["tb"].running():
            jobs["tb"].start([PYTHON, "-m", "tensorboard.main",
                              "--logdir", _rel(log_dir()), "--port", "6006"])
        webbrowser.open("http://localhost:6006")


def handle_infer(event, vals, jobs, window, ckpt_map, state):
    player = state["player"]
    if event == "-CKPT-REFRESH-":
        current_ckpt = vals.get("-CKPT-")
        current_spk = vals.get("-SPK-")
        saved = load_settings().get("infer", {})
        ckpt_map.clear()
        ckpt_map.update({ckpt_label(p): p for p in list_ckpts()})
        window["-CKPT-"].update(values=list(ckpt_map),
                                value=keep_selection(ckpt_map, current_ckpt, saved.get("-CKPT-")))
        window["-SPK-"].update(values=list_speakers(),
                               value=keep_selection(list_speakers(), current_spk, saved.get("-SPK-")))
    elif event == "-RUN-":
        if jobs["infer"].running():
            return
        if not vals["-IN-"] or not os.path.exists(vals["-IN-"]):
            jobs["infer"].buf.clear()
            jobs["infer"].buf.feed("请先选择有效的输入音频。\n")
            return
        if vals["-CKPT-"] not in ckpt_map:
            jobs["infer"].buf.clear()
            jobs["infer"].buf.feed("请先选择模型（训练出 G_*.pth 后点刷新）。\n")
            return
        if not vals["-SPK-"]:
            jobs["infer"].buf.clear()
            jobs["infer"].buf.feed("请先选择说话人。\n")
            return
        model = ckpt_map[vals["-CKPT-"]]
        step_n = _step_of(model, "G")
        base = os.path.splitext(os.path.basename(vals["-IN-"]))[0]
        # 官方输出命名固定，这里指定输出目录，转换后按官方命名找回最新文件
        state["out_dir"] = OUT_DIR
        state["out_base"] = base
        state["out_before"] = {
            p: os.path.getmtime(p)
            for p in glob.glob(os.path.join(OUT_DIR, "*"))
        }
        out_placeholder = os.path.join(OUT_DIR, f"{base}.wav")
        window["-OUTPATH-"].update("转换中...")
        jobs["infer"].buf.clear()
        jobs["infer"].start(infer_cmd(vals, model, out_placeholder))
    else:
        handle_player_preset(event, vals, window, state, player)


def handle_player_preset(event, vals, window, state, player):
    if event == "-PLAY-":
        out = state.get("out")
        if out and os.path.exists(out):
            if state.get("loaded") != out:
                player.load(out)
                state["loaded"] = out
            player.set_volume(vals["-VOL-"])
            player.play()
    elif event == "-PAUSE-":
        player.pause()
    elif event == "-PSTOP-":
        player.stop()
    elif event == "-SEEK-":
        if player.total:
            player.seek(vals["-SEEK-"] / 1000.0)
    elif event == "-VOL-":
        player.set_volume(vals["-VOL-"])
        window["-VOLPCT-"].update(f"{int(vals['-VOL-'])}%")
    elif event == "-PRESET-REFRESH-":
        window["-PRESET-"].update(values=list_presets())
    elif event == "-PRESET-LOAD-":
        if vals["-PRESET-"]:
            load_preset(vals["-PRESET-"], window)
            window["-PRESET-MSG-"].update(f"已加载 {vals['-PRESET-']}")
    elif event == "-PRESET-SAVE-":
        name = vals["-PRESET-NAME-"] or vals["-PRESET-"]
        try:
            saved = save_preset(name, vals)
        except ValueError as e:
            window["-PRESET-MSG-"].update(str(e), text_color="red")
            return
        window["-PRESET-"].update(values=list_presets(), value=saved)
        window["-PRESET-MSG-"].update(f"已保存 {saved}", text_color="yellow")
    elif event == "-PRESET-DEL-":
        if vals["-PRESET-"]:
            delete_preset(vals["-PRESET-"])
            window["-PRESET-"].update(values=list_presets(), value="")
            window["-PRESET-MSG-"].update("已删除")


def main():
    window, chart = build_window()
    jobs = {k: Job() for k in ("ds", "train", "diff", "tb", "infer")}
    jobs["tb"].buf = jobs["train"].buf   # TensorBoard 输出并入训练日志
    ckpt_map = {ckpt_label(p): p for p in list_ckpts()}
    state = {"out": None, "out_dir": None, "out_base": None, "out_before": set(),
             "player": Player(), "loaded": None, "chart": chart,
             "last_chart": 0.0, "target_step": 0, "patch_after_ds": False}
    state["player"].set_volume(load_settings().get("volume", 100))
    if has_project():
        refresh_project_ui(window, ckpt_map, state)
    apply_infer_settings(window)
    do_scan({"-DS-SCANDIR-": window["-DS-SCANDIR-"].get()}, window)

    while True:
        event, vals = window.read(timeout=100)
        if event == sg.WIN_CLOSED:
            break

        handle_project(event, vals, window, ckpt_map, state)
        handle_dataset(event, vals, jobs, window, state)
        handle_config(event, vals, window)
        handle_train(event, vals, jobs, window, state)
        handle_infer(event, vals, jobs, window, ckpt_map, state)
        if vals:
            current_infer = infer_settings(vals)
            if current_infer != state.get("infer_last"):
                save_infer_settings(vals)
                state["infer_last"] = current_infer

        jobs["ds"].drain_to(window, "-DS-LOG-")
        jobs["train"].drain_to(window, "-TRAIN-LOG-")
        jobs["diff"].drain_to(window, "-DIFF-LOG-")
        jobs["infer"].drain_to(window, "-INF-LOG-")

        # 推理完成：按官方命名找回新生成的输出文件
        if not jobs["infer"].done_q.empty():
            rc = jobs["infer"].done_q.get()
            audio = updated_audio_files(state["out_dir"] or OUT_DIR, state.get("out_before", {}))
            out = max(audio, key=os.path.getmtime) if audio else None
            if rc == 0 and out and os.path.exists(out):
                state["out"] = out
                window["-OUTPATH-"].update(f"完成: {out}")
                state["player"].load(out)
                state["loaded"] = out
                state["player"].set_volume(vals["-VOL-"])
                state["player"].play()
            else:
                window["-OUTPATH-"].update("转换失败，看日志。")
            # 推理后刷新模型列表(可能训练同时产出新ckpt)
            current_ckpt = vals.get("-CKPT-")
            ckpt_map.clear()
            ckpt_map.update({ckpt_label(p): p for p in list_ckpts()})
            window["-CKPT-"].update(values=list(ckpt_map),
                                    value=keep_selection(ckpt_map, current_ckpt, infer_saved("-CKPT-")))

        # ds 完成：若刚生成过配置，做工程化修正；刷新工程相关界面
        while not jobs["ds"].done_q.empty():
            rc = jobs["ds"].done_q.get()
            if rc == 0 and state.get("patch_after_ds"):
                patch_configs_for_project()
                state["patch_after_ds"] = False
                refresh_project_ui(window, ckpt_map, state)
            window["-DS-PIPE-MSG-"].update(
                "完成。可去训练页开始训练。" if rc == 0
                else f"中止/失败(退出码 {rc})，请看日志。",
                text_color="green" if rc == 0 else "red")
        for k in ("train", "diff", "tb"):
            while not jobs[k].done_q.empty():
                jobs[k].done_q.get()

        update_player_ui(window, state)

        # 训练中每5s刷新 tfevents：曲线刷新 + 目标step自动停
        if vals and jobs["train"].running() and time.time() - state["last_chart"] >= 5:
            state["last_chart"] = time.time()
            tags = chart.reload(log_dir())
            if vals.get("-CHART-AUTO-"):
                tag = vals["-CHART-TAG-"] or ("loss/g/total" if "loss/g/total" in tags else "")
                if tag:
                    if tag not in (window["-CHART-TAG-"].Values or []):
                        window["-CHART-TAG-"].update(values=tags, value=tag)
                    window["-CHART-MSG-"].update(chart.plot(tag, vals["-CHART-LOG-"]))
            target = state.get("target_step", 0)
            cur = max((s[-1] for s, _ in chart.scalars.values() if s), default=0)
            if target and cur >= target:
                state["target_step"] = 0
                jobs["train"].stop()
                window["-TRAIN-MSG-"].update(
                    f"已达 {cur} step（目标 {target}），自动停止训练。", text_color="green")

    state["player"].stop()
    if vals:
        settings = load_settings()
        settings.update({
            "volume": int(vals["-VOL-"]),
            "project": PROJ["name"],
            "device": vals["-DEV-"],
            "tensorboard": bool(vals["-TR-TB-"]),
            "chart_auto": bool(vals["-CHART-AUTO-"]),
            "chart_log": bool(vals["-CHART-LOG-"]),
        })
        settings["infer"] = infer_settings(vals)
        save_settings(settings)
    for j in jobs.values():
        if j.running():
            j.proc.terminate()
    window.close()


if __name__ == "__main__":
    main()



























