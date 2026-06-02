# -*- coding: utf-8 -*-
"""数据集预处理标签页 UI"""
import os
import dearpygui.dearpygui as dpg
from . import config, backend, ui_dialogs
from .ui_log import clear_job_log, add_log_panel


def _fmt_dur(sec):
    s = int(sec)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


DATASET_SETTING_KEYS = (
    "ds_scandir",
    "ds_outdir",
    "rs_sr",
    "rs_skip_loudnorm",
    "rs_numproc",
    "ds_encoder",
    "ds_volaug",
    "ds_reuse_config",
    "ds_eta_dir",
    "ds_f0method",
    "ds_numproc",
    "ds_usediff",
)


def _saved_value(settings, key, default):
    return settings.get("dataset", {}).get(key, default)


def _save_dataset_setting(key, value):
    settings = config.load_settings()
    settings.setdefault("dataset", {})[key] = value
    config.save_settings(settings)


def _save_item(key):
    def callback(sender=None, app_data=None, user_data=None):
        _save_dataset_setting(key, dpg.get_value(key))
    return callback


def _save_all_dataset_settings():
    settings = config.load_settings()
    dataset = settings.setdefault("dataset", {})
    for key in DATASET_SETTING_KEYS:
        if dpg.does_item_exist(key):
            dataset[key] = dpg.get_value(key)
    config.save_settings(settings)


def _browse_scandir():
    p = ui_dialogs.pick_directory("选择源数据集目录")
    if p:
        dpg.set_value("ds_scandir", p)
        _save_dataset_setting("ds_scandir", p)


def _browse_outdir():
    p = ui_dialogs.pick_directory("选择 44k 输出目录")
    if p:
        dpg.set_value("ds_outdir", p)
        _save_dataset_setting("ds_outdir", p)


def _browse_eta_dir():
    p = ui_dialogs.pick_directory("选择多说话人语料目录")
    if p:
        dpg.set_value("ds_eta_dir", p)
        _save_dataset_setting("ds_eta_dir", p)


def _sync_filelist(state):
    if not state["current_project"]:
        dpg.set_value("ds_pipe_msg", "请先创建/选择工程。")
        return
    _save_all_dataset_settings()
    proj = state["current_project"]
    n = config.rewrite_filelist_to_outdir(proj)
    if n:
        dpg.set_value("ds_pipe_msg", f"已更新 filelist {n} 行 → {config.dataset_44k_dir()}")
    else:
        dpg.set_value("ds_pipe_msg", "未找到 filelist，请先生成配置。")


def _eta_proj_path():
    return os.path.join(backend.ROOT, "pretrain", "eta_wavlm_proj.pt")


def _eta_proj_status():
    return "投影已存在" if os.path.exists(_eta_proj_path()) else "尚未拟合投影"


def _whisper_path():
    for name in ("large-v3.pt", "large-v2.pt"):
        p = os.path.join(backend.ROOT, "pretrain", name)
        if os.path.exists(p):
            return p
    return None


def _whisper_status():
    p = _whisper_path()
    return f"已就位 ({os.path.basename(p)})" if p else "缺 large-v3.pt"


def _refresh_encoder_sections(enc):
    """按所选编码器显示/隐藏其专属区块。"""
    if dpg.does_item_exist("ds_eta_section"):
        dpg.configure_item("ds_eta_section", show=(enc == "etawavlmlarge"))
    if dpg.does_item_exist("ds_whisper_section"):
        dpg.configure_item("ds_whisper_section", show=(enc == "whisper+contentvec"))


def _on_encoder_change(sender=None, app_data=None, user_data=None):
    enc = dpg.get_value("ds_encoder")
    _save_dataset_setting("ds_encoder", enc)
    _refresh_encoder_sections(enc)


def _run_whisper_download(state):
    if _whisper_path() and os.path.basename(_whisper_path()) == "large-v3.pt":
        dpg.set_value("ds_whisper_msg", "large-v3.pt 已存在")
        return
    dpg.set_value("ds_whisper_msg", "下载中…(见日志)")
    state["jobs"]["ds"].start(backend.whisper_download_cmd())


def create_dataset_tab(state):
    """创建数据集预处理标签页"""
    cpu = os.cpu_count() or 4
    settings = config.load_settings()
    default_scan_dir = os.path.relpath(backend.DATASET_RAW, backend.ROOT)
    default_out_dir = os.path.join("dataset", "44k")

    with dpg.child_window(tag="tab_dataset"):
        # 数据集概览
        with dpg.collapsing_header(label="数据集概览", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("源目录:")
                dpg.add_input_text(tag="ds_scandir",
                                   default_value=_saved_value(settings, "ds_scandir", default_scan_dir),
                                   width=400, callback=_save_item("ds_scandir"))
                dpg.add_button(label="浏览", callback=lambda: _browse_scandir())
                dpg.add_button(label="扫描", callback=lambda: _scan_dataset(state))
                dpg.add_text("", tag="ds_scan_msg", color=(255, 255, 100))

            dpg.add_text("训练音频放到 dataset_raw/{说话人}/*.wav，每个说话人一个子目录。", color=(150, 150, 150))
            dpg.add_spacer(height=5)

            with dpg.table(tag="ds_table", header_row=True, borders_innerH=True, borders_outerH=True,
                           borders_innerV=True, borders_outerV=True, row_background=True):
                dpg.add_table_column(label="说话人", width_fixed=True, init_width_or_weight=200)
                dpg.add_table_column(label="文件数", width_fixed=True, init_width_or_weight=100)
                dpg.add_table_column(label="时长", width_fixed=True, init_width_or_weight=120)

        dpg.add_spacer(height=10)

        # 第一步：重采样
        with dpg.collapsing_header(label="第一步：重采样到 44.1kHz (resample.py)", default_open=True):
            dpg.add_text("dataset_raw → dataset/44k，自动去静音；未跳过时会逐段 peak normalize。")
            with dpg.group(horizontal=True):
                dpg.add_text("输出目录:")
                dpg.add_input_text(tag="ds_outdir",
                                   default_value=_saved_value(settings, "ds_outdir", default_out_dir),
                                   width=400, callback=_save_item("ds_outdir"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("重采样后的 wav 与提取的特征(.soft.pt/.f0.npy/.spec.pt)都存这里，\n"
                                 "训练时每步从此读取。项目在机械硬盘时，可改为 SSD 上的绝对路径\n"
                                 "(如 D:\\ssd\\dataset_44k)以加速训练。所有工程共用此目录，\n"
                                 "切换工程重新预处理会覆盖。留空=项目内 dataset/44k。")
                dpg.add_button(label="浏览", callback=lambda: _browse_outdir())
                dpg.add_button(label="同步filelist", callback=lambda: _sync_filelist(state))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("把当前工程的 train.txt/val.txt 里每行音频路径的目录前缀，\n"
                                 "替换为上面填的输出目录(保留原有 train/val 划分与说话人)。\n"
                                 "用于：已手动把数据集挪到新目录、但不想重抽特征时，\n"
                                 "一键让训练 filelist 指向新位置。\n"
                                 "前提：新目录里是「输出目录/说话人/*.wav」结构。")
            with dpg.group(horizontal=True):
                dpg.add_text("目标采样率:")
                dpg.add_input_int(tag="rs_sr", default_value=_saved_value(settings, "rs_sr", 44100),
                                  width=100, step=0, callback=_save_item("rs_sr"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("官方默认44100Hz，不建议修改。\n范围: 22050-48000")
                dpg.add_checkbox(label="跳过响度归一(--skip_loudnorm)", tag="rs_skip_loudnorm",
                                 default_value=_saved_value(settings, "rs_skip_loudnorm", False),
                                 callback=_save_item("rs_skip_loudnorm"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("源音频已经做过响度均衡时建议勾选。\n可避免 resample.py 再把每段 peak 拉满。")
                dpg.add_text("并行进程:")
                dpg.add_drag_int(tag="rs_numproc", default_value=_saved_value(settings, "rs_numproc", 0),
                                 min_value=0, max_value=cpu, width=120, callback=_save_item("rs_numproc"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(f"0=脚本自动，上限8。CPU核心数: {cpu}")
                dpg.add_button(label="重采样", callback=lambda: _run_resample(state))

        dpg.add_spacer(height=10)

        # 第二步：生成配置
        with dpg.collapsing_header(label="第二步：生成配置 (preprocess_flist_config.py)", default_open=True):
            dpg.add_text("生成 config.json + diffusion.yaml 到工程目录。")
            with dpg.group(horizontal=True):
                dpg.add_text("内容编码器:")
                dpg.add_combo(config.SPEECH_ENCODERS, tag="ds_encoder",
                              default_value=_saved_value(settings, "ds_encoder", "vec768l12"),
                              width=150, callback=_on_encoder_change)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("内容特征提取器，须与训练页保持一致。\n"
                                 "· vec768l12：默认，唱歌综合最优(ssl_dim=768)\n"
                                 "· vec768l12mix：预处理写 2304 维三层 ContentVec，模型内合并到 ssl_dim=768\n"
                                 "· wavlmlarge：WavLM-Large 第6层，解耦更强、咬字更准\n"
                                 "  (ssl_dim 自动设为 1024，需放置 pretrain/WavLM-Large.pt)\n"
                                 "· whisper+contentvec：双编码器(ssl_dim=2048)，质量上限最高、单说话人推荐\n"
                                 "  需 pretrain/large-v3.pt(python download_whisper.py)，预处理较慢\n"
                                 "· cnhubertlarge/whisper-ppg：偏说话场景\n"
                                 "切换编码器后必须重新提取特征并重训。")
                dpg.add_checkbox(label="音量增强(vol_aug)", tag="ds_volaug",
                                 default_value=_saved_value(settings, "ds_volaug", False),
                                 callback=_save_item("ds_volaug"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("训练时随机调整音量，增强鲁棒性")
                dpg.add_checkbox(label="复用已有配置", tag="ds_reuse_config",
                                 default_value=_saved_value(settings, "ds_reuse_config", True),
                                 callback=_save_item("ds_reuse_config"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("工程内已存在 config.json/diffusion.yaml 时，以其为基底，\n"
                                 "仅更新说话人列表与编码器相关字段，保留已在训练页调好的\n"
                                 "batch_size/学习率/判别器/增强等超参，无需重新设置。\n"
                                 "首次生成或文件不存在时此项无影响。")
                dpg.add_button(label="生成配置", callback=lambda: _run_preconfig(state))

        dpg.add_spacer(height=10)

        # 可选：拟合 Eta-WavLM 去说话人投影（仅 etawavlmlarge 需要，按编码器条件显示）
        with dpg.collapsing_header(label="拟合 Eta-WavLM 去说话人投影 (eta_wavlm_fit.py)",
                                   tag="ds_eta_section", default_open=True, show=False):
            dpg.add_text("仅当内容编码器选 etawavlmlarge 时需要。在多说话人语料上拟合一次，\n"
                         "生成 pretrain/eta_wavlm_proj.pt（全局复用、跨工程共享），\n"
                         "必须在「第三步：提取特征」之前完成，否则提取会因缺投影而报错。",
                         color=(150, 150, 150))
            with dpg.group(horizontal=True):
                dpg.add_text("多说话人语料目录:")
                dpg.add_input_text(tag="ds_eta_dir",
                                   default_value=_saved_value(settings, "ds_eta_dir", default_scan_dir),
                                   width=360, callback=_save_item("ds_eta_dir"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("含多个说话人子目录的 wav/flac/mp3 语料。\n"
                                 "投影刻画「说话人信息在 WavLM 空间的位置」，与数据集无关，\n"
                                 "建议用尽量多的说话人；用单说话人歌声集拟合是病态的。")
                dpg.add_button(label="浏览", callback=lambda: _browse_eta_dir())
                dpg.add_button(label="拟合投影", callback=lambda: _run_eta_fit(state))
                dpg.add_text(_eta_proj_status(), tag="ds_eta_msg", color=(255, 255, 100))

        dpg.add_spacer(height=10)

        # Whisper 权重（仅 whisper+contentvec 需要，按编码器条件显示）
        with dpg.collapsing_header(label="下载 Whisper 权重 (whisper+contentvec 所需)",
                                   tag="ds_whisper_section", default_open=True, show=False):
            dpg.add_text("whisper+contentvec 需要 pretrain/large-v3.pt(~3GB)，组合编码器的 Whisper 分支用它。",
                         color=(150, 150, 150))
            with dpg.group(horizontal=True):
                dpg.add_button(label="下载 large-v3", callback=lambda: _run_whisper_download(state))
                dpg.add_text(_whisper_status(), tag="ds_whisper_msg", color=(255, 255, 100))

        dpg.add_spacer(height=10)

        # 第三步：提取特征
        with dpg.collapsing_header(label="第三步：提取特征+f0 (preprocess_hubert_f0.py)", default_open=True):
            dpg.add_text("提取内容编码和音高特征。")
            with dpg.group(horizontal=True):
                dpg.add_text("f0 提取器:")
                dpg.add_combo(config.F0_METHODS, tag="ds_f0method",
                              default_value=_saved_value(settings, "ds_f0method", "rmvpe"),
                              width=120, callback=_save_item("ds_f0method"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("唱歌推荐 rmvpe，精度高\ncrepe 也可，但慢")
                dpg.add_text("并行进程:")
                dpg.add_drag_int(tag="ds_numproc", default_value=_saved_value(settings, "ds_numproc", 1),
                                 min_value=0, max_value=cpu, width=150, callback=_save_item("ds_numproc"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(f"多进程加速，0=自动。CPU核心数: {cpu}")
                dpg.add_checkbox(label="同时提取扩散特征(--use_diff)", tag="ds_usediff",
                                 default_value=_saved_value(settings, "ds_usediff", True),
                                 callback=_save_item("ds_usediff"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("勾选后才能训练浅扩散模型")
                dpg.add_button(label="提取特征", callback=lambda: _run_hubert(state))

        dpg.add_spacer(height=10)

        # 一键全流程
        with dpg.collapsing_header(label="一键全流程", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="一键预处理", callback=lambda: _run_all(state), width=120)
                dpg.add_button(label="停止", callback=lambda: _stop_ds(state))
                dpg.add_text("依次执行 重采样→生成配置→提取特征。", color=(150, 150, 150))
                dpg.add_text("", tag="ds_pipe_msg", color=(255, 255, 100))

        dpg.add_spacer(height=10)

        # 日志
        with dpg.group(horizontal=True):
            dpg.add_text("执行日志:")
            dpg.add_button(label="复制全部", callback=lambda: _copy_log(state, "ds"))
            dpg.add_button(label="清除内容", callback=lambda: clear_job_log(state, "ds", "ds_log", "ds_pipe_msg"))
        add_log_panel("ds_log", "ds_log_win", 300)

    # 按当前编码器初始化专属区块的可见性
    _refresh_encoder_sections(dpg.get_value("ds_encoder"))


def _scan_dataset(state):
    scan_dir = dpg.get_value("ds_scandir")
    _save_dataset_setting("ds_scandir", scan_dir)
    rows, n, dur = config.scan_dataset(scan_dir)
    dpg.delete_item("ds_table", children_only=True, slot=1)
    for spk, files, d in rows:
        with dpg.table_row(parent="ds_table"):
            dpg.add_text(spk)
            dpg.add_text(str(files))
            dpg.add_text(_fmt_dur(d))
    if rows:
        dpg.set_value("ds_scan_msg", f"{len(rows)} 个说话人，共 {n} 个文件，总时长 {_fmt_dur(dur)}")
    else:
        dpg.set_value("ds_scan_msg", "未找到说话人子目录或音频文件。")


def _run_resample(state):
    if not state["current_project"]:
        dpg.set_value("ds_pipe_msg", "请先创建/选择工程。")
        return
    _save_all_dataset_settings()
    sr = dpg.get_value("rs_sr")
    skip_loudnorm = dpg.get_value("rs_skip_loudnorm")
    num_proc = dpg.get_value("rs_numproc")
    scan_dir = dpg.get_value("ds_scandir")
    in_dir = scan_dir if os.path.isabs(scan_dir) else os.path.join(backend.ROOT, scan_dir)
    cmd = backend.resample_cmd(sr, in_dir, config.dataset_44k_dir(), skip_loudnorm, num_proc)
    state["jobs"]["ds"].start(cmd)


def _run_preconfig(state):
    if not state["current_project"]:
        dpg.set_value("ds_pipe_msg", "请先创建/选择工程。")
        return
    _save_all_dataset_settings()
    proj = state["current_project"]
    encoder = dpg.get_value("ds_encoder")
    vol_aug = dpg.get_value("ds_volaug")
    reuse_config = dpg.get_value("ds_reuse_config")
    train_list = os.path.join(config.filelist_dir(proj), "train.txt")
    val_list = os.path.join(config.filelist_dir(proj), "val.txt")
    cfg_out = config.config_path(proj)
    diff_out = config.diff_config_path(proj)
    cmd = backend.config_cmd(encoder, vol_aug, config.dataset_44k_dir(), train_list, val_list, cfg_out, diff_out, reuse_config)
    state["patch_after_ds"] = True
    state["jobs"]["ds"].start(cmd)


def _run_eta_fit(state):
    eta_dir = dpg.get_value("ds_eta_dir")
    if not eta_dir:
        dpg.set_value("ds_eta_msg", "请先填多说话人语料目录。")
        return
    _save_all_dataset_settings()
    in_dir = eta_dir if os.path.isabs(eta_dir) else os.path.join(backend.ROOT, eta_dir)
    if not os.path.isdir(in_dir):
        dpg.set_value("ds_eta_msg", "目录不存在。")
        return
    dpg.set_value("ds_eta_msg", "拟合中…")
    state["jobs"]["ds"].start(backend.eta_fit_cmd(in_dir))


def _run_hubert(state):
    if not state["current_project"]:
        dpg.set_value("ds_pipe_msg", "请先创建/选择工程。")
        return
    _save_all_dataset_settings()
    proj = state["current_project"]
    f0_method = dpg.get_value("ds_f0method")
    num_proc = dpg.get_value("ds_numproc")
    use_diff = dpg.get_value("ds_usediff")
    cfg = config.config_path(proj)
    diff_cfg = config.diff_config_path(proj)
    cmd = backend.hubert_cmd(f0_method, num_proc, use_diff, config.dataset_44k_dir(), cfg, diff_cfg)
    state["jobs"]["ds"].start(cmd)


def _run_all(state):
    if not state["current_project"]:
        dpg.set_value("ds_pipe_msg", "请先创建/选择工程。")
        return
    _save_all_dataset_settings()
    state["jobs"]["ds"].buf.clear()
    state["patch_after_ds"] = True
    dpg.set_value("ds_pipe_msg", "一键全流程运行中…")

    proj = state["current_project"]
    sr = dpg.get_value("rs_sr")
    skip_loudnorm = dpg.get_value("rs_skip_loudnorm")
    rs_num_proc = dpg.get_value("rs_numproc")
    scan_dir = dpg.get_value("ds_scandir")
    in_dir = scan_dir if os.path.isabs(scan_dir) else os.path.join(backend.ROOT, scan_dir)
    encoder = dpg.get_value("ds_encoder")
    vol_aug = dpg.get_value("ds_volaug")
    reuse_config = dpg.get_value("ds_reuse_config")
    f0_method = dpg.get_value("ds_f0method")
    num_proc = dpg.get_value("ds_numproc")
    use_diff = dpg.get_value("ds_usediff")

    train_list = os.path.join(config.filelist_dir(proj), "train.txt")
    val_list = os.path.join(config.filelist_dir(proj), "val.txt")
    cfg_out = config.config_path(proj)
    diff_out = config.diff_config_path(proj)

    steps = [
        ("重采样", backend.resample_cmd(sr, in_dir, config.dataset_44k_dir(), skip_loudnorm, rs_num_proc)),
        ("生成配置", backend.config_cmd(encoder, vol_aug, config.dataset_44k_dir(), train_list, val_list, cfg_out, diff_out, reuse_config)),
        ("提取特征+f0", backend.hubert_cmd(f0_method, num_proc, use_diff, config.dataset_44k_dir(), cfg_out, diff_out)),
    ]
    state["jobs"]["ds"].start_chain(steps)


def _stop_ds(state):
    msg = state["jobs"]["ds"].stop()
    state["jobs"]["ds"].buf.feed(msg)


def _copy_log(state, job_key):
    """复制日志到剪贴板"""
    text = state["jobs"][job_key].buf.snapshot()
    try:
        import pyperclip
        pyperclip.copy(text)
        dpg.set_value("ds_pipe_msg", "日志已复制到剪贴板")
    except ImportError:
        # pyperclip 不可用时，至少把文本放到系统剪贴板（Windows）
        try:
            import subprocess
            subprocess.run(['clip'], input=text.encode('utf-16le'), check=True)
            dpg.set_value("ds_pipe_msg", "日志已复制到剪贴板")
        except Exception as e:
            dpg.set_value("ds_pipe_msg", f"复制失败: {e}")
