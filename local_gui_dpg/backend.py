# -*- coding: utf-8 -*-
"""后端逻辑：Job/TermBuffer/Player/命令构造，复用自 local_gui.py"""
import os
import re
import queue
import codecs
import threading
import subprocess
import sounddevice as sd
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, "venv", "python.exe")
DATASET_RAW = os.path.join(ROOT, "dataset_raw")
DATASET_44K = os.path.join(ROOT, "dataset", "44k")

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
            try:
                # Windows: 使用 taskkill 强制结束进程树
                import platform
                if platform.system() == "Windows":
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.proc.pid)],
                                   capture_output=True, timeout=5)
                else:
                    self.proc.terminate()
                    self.proc.wait(timeout=5)
            except Exception as e:
                self.buf.feed(f"[警告] 停止进程时出错: {e}\n")
                try:
                    self.proc.kill()
                except Exception:
                    pass
            return "已发送停止信号。\n"
        return "当前没有正在运行的进程。\n"


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


def _py(script):
    return [PYTHON, os.path.join(ROOT, script)]


def _rel(p):
    p = os.path.abspath(p)
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def resample_cmd(sr, in_dir, out_dir, skip_loudnorm=False, num_proc=0):
    cmd = _py("resample.py") + [
        "--in_dir", _rel(in_dir),
        "--out_dir2", _rel(out_dir),
        "--sr2", str(int(sr))]
    if skip_loudnorm:
        cmd += ["--skip_loudnorm"]
    if num_proc:
        cmd += ["--num_processes", str(int(num_proc))]
    return cmd


def config_cmd(encoder, vol_aug, source_dir, train_list, val_list, config_out, diff_config_out, reuse_config=False):
    cmd = _py("preprocess_flist_config.py") + [
        "--source_dir", _rel(source_dir),
        "--speech_encoder", encoder,
        "--train_list", _rel(train_list),
        "--val_list", _rel(val_list),
        "--config_out", _rel(config_out),
        "--diff_config_out", _rel(diff_config_out)]
    if vol_aug:
        cmd += ["--vol_aug"]
    if reuse_config:
        cmd += ["--reuse_config"]
    return cmd


def eta_fit_cmd(in_dir, out_path=None, output_layer=6):
    out_path = out_path or os.path.join(ROOT, "pretrain", "eta_wavlm_proj.pt")
    return _py("eta_wavlm_fit.py") + [
        "--in_dir", _rel(in_dir),
        "--out", _rel(out_path),
        "--output_layer", str(int(output_layer))]


def whisper_download_cmd(model="large-v3"):
    return _py("download_whisper.py") + ["--model", model]


def hubert_cmd(f0_method, num_proc, use_diff, in_dir, config_path, diff_config_path):
    cmd = _py("preprocess_hubert_f0.py") + [
        "--in_dir", _rel(in_dir),
        "--config", _rel(config_path),
        "--diff-config", _rel(diff_config_path),
        "--f0_predictor", f0_method]
    if num_proc:
        cmd += ["--num_processes", str(int(num_proc))]
    if use_diff:
        cmd += ["--use_diff"]
    return cmd


def train_cmd(config_path, model_name):
    return _py("train.py") + ["-c", _rel(config_path), "-m", model_name]


def diff_train_cmd(diff_config_path):
    return _py("train_diff.py") + ["-c", _rel(diff_config_path)]


def infer_cmd(model_path, config_path, input_path, output_dir, trans, spk, slice_db,
              noise_scale, pad, clip, f0_method, device, auto_f0, enhance, feature_retrieval,
              cluster_path, cluster_ratio, shallow_diff, k_step, second_enc, lea, diff_model, diff_config):
    dev = "cuda:0" if device == "GPU" else "cpu"
    cmd = _py("inference_main.py") + [
        "-m", _rel(model_path),
        "-c", _rel(config_path),
        "-ip", input_path,
        "-op", output_dir,
        "-t", str(int(trans)),
        "-s", spk,
        "-sd", str(int(slice_db)),
        "-ns", str(noise_scale),
        "-p", str(pad),
        "-cl", str(clip),
        "-f0p", f0_method,
        "-d", dev,
        "-wf", "wav",
        "-lea", str(lea)]
    if auto_f0:
        cmd += ["-a"]
    if enhance:
        cmd += ["-eh"]
    if feature_retrieval:
        cmd += ["-fr"]
    if cluster_path and os.path.exists(cluster_path) and cluster_ratio > 0:
        cmd += ["-cm", _rel(cluster_path), "-cr", str(cluster_ratio)]
    if shallow_diff and diff_model and os.path.exists(diff_model):
        cmd += ["-shd", "-dm", _rel(diff_model), "-dc", _rel(diff_config), "-ks", str(int(k_step))]
        if second_enc:
            cmd += ["-se"]
    return cmd


def cluster_train_cmd(dataset_dir, output_dir, n_clusters, use_gpu):
    """训练 KMeans 聚类模型"""
    cmd = _py("cluster/train_cluster.py") + [
        "--dataset", _rel(dataset_dir),
        "--output", _rel(output_dir)]
    if use_gpu:
        cmd += ["--gpu"]
    return cmd


def index_train_cmd(dataset_dir, config_path, output_dir):
    """训练特征检索索引"""
    return _py("train_index.py") + [
        "--root_dir", _rel(dataset_dir),
        "-c", _rel(config_path),
        "--output_dir", _rel(output_dir)]
