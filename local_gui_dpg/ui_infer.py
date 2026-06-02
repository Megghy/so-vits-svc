# -*- coding: utf-8 -*-
"""推理标签页 UI"""
import os
import glob
import dearpygui.dearpygui as dpg
from . import config, backend
from .ui_log import clear_job_log


INFER_SETTING_KEYS = [
    "inf_input",
    "inf_ckpt",
    "inf_spk",
    "inf_device",
    "inf_trans",
    "inf_f0method",
    "inf_autof0",
    "inf_slicedb",
    "inf_noise",
    "inf_shallow",
    "inf_kstep",
    "inf_second_enc",
    "inf_diff_model",
    "inf_enhance",
    "inf_feature_retrieval",
    "inf_cluster",
    "inf_cluster_ratio",
    "inf_pad",
    "inf_clip",
    "inf_lea",
    "inf_preset",
]


def _saved_value(settings, key, default):
    return settings.get("infer", {}).get(key, default)


def _select_value(labels, current, saved):
    if current in labels:
        return current
    if saved in labels:
        return saved
    return labels[0] if labels else ""


def _save_infer_setting(key, value):
    settings = config.load_settings()
    settings.setdefault("infer", {})[key] = value
    config.save_settings(settings)


def _save_item(key):
    if dpg.does_item_exist(key):
        _save_infer_setting(key, dpg.get_value(key))


def _save_all_infer_settings():
    settings = config.load_settings()
    infer = settings.setdefault("infer", {})
    for key in INFER_SETTING_KEYS:
        if dpg.does_item_exist(key):
            infer[key] = dpg.get_value(key)
    config.save_settings(settings)


def create_infer_tab(state):
    """创建推理标签页"""
    # 加载历史输入文件
    settings = config.load_settings()
    recent_inputs = settings.get("recent_inputs", [])
    saved_input = _saved_value(settings, "inf_input", recent_inputs[0] if recent_inputs else "")
    if saved_input and saved_input not in recent_inputs:
        recent_inputs.insert(0, saved_input)

    with dpg.child_window(tag="tab_infer"):
        # 常用设置
        with dpg.collapsing_header(label="常用设置", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("输入音频:")
                dpg.add_combo(
                    recent_inputs,
                    tag="inf_input",
                    width=500,
                    default_value=saved_input,
                    callback=lambda s=None, a=None, u=None: _save_item("inf_input"))
                dpg.add_button(label="浏览", callback=lambda: _browse_input_file(state))
            with dpg.group(horizontal=True):
                dpg.add_text("模型:")
                dpg.add_combo([], tag="inf_ckpt", width=400,
                              callback=lambda s=None, a=None, u=None: _save_item("inf_ckpt"))
                dpg.add_button(label="刷新", callback=lambda: _refresh_ckpts(state))
            with dpg.group(horizontal=True):
                dpg.add_text("说话人:")
                dpg.add_combo([], tag="inf_spk", width=200,
                              callback=lambda s=None, a=None, u=None: _save_item("inf_spk"))
                dpg.add_text("设备:")
                dpg.add_combo(
                    ["GPU", "CPU"],
                    tag="inf_device",
                    default_value=_saved_value(settings, "inf_device", settings.get("device", "GPU")),
                    width=80,
                    callback=lambda s=None, a=None, u=None: _save_item("inf_device"))
            with dpg.group(horizontal=True):
                dpg.add_text("变调(半音):")
                dpg.add_drag_int(
                    tag="inf_trans",
                    default_value=_saved_value(settings, "inf_trans", 0),
                    min_value=-24,
                    max_value=24,
                    width=400,
                    callback=lambda s=None, a=None, u=None: _save_item("inf_trans"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("音高调整，正数升调，负数降调。\n范围: -24 ~ +24 半音")

        dpg.add_spacer(height=10)

        # 音高 / f0
        with dpg.collapsing_header(label="音高 / f0", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("f0 提取器:")
                dpg.add_combo(
                    config.F0_METHODS,
                    tag="inf_f0method",
                    default_value=_saved_value(settings, "inf_f0method", "rmvpe"),
                    width=120,
                    callback=lambda s=None, a=None, u=None: _save_item("inf_f0method"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("rmvpe: 唱歌推荐，精度高\ncrepe: 精度高但慢\npm/dio/harvest: 快但精度低")
                dpg.add_checkbox(
                    label="自动预测f0(说话用,唱歌关)",
                    tag="inf_autof0",
                    default_value=_saved_value(settings, "inf_autof0", False),
                    callback=lambda s=None, a=None, u=None: _save_item("inf_autof0"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("语音转换时开启，唱歌转换必须关闭否则跑调")
            with dpg.group(horizontal=True):
                dpg.add_text("切片阈值(dB):")
                dpg.add_drag_int(
                    tag="inf_slicedb",
                    default_value=_saved_value(settings, "inf_slicedb", -40),
                    min_value=-60,
                    max_value=-20,
                    width=200,
                    callback=lambda s=None, a=None, u=None: _save_item("inf_slicedb"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("静音切片阈值。\n-40: 默认\n-30: 嘈杂音频\n-50: 保留呼吸声")
                dpg.add_text("噪声scale:")
                dpg.add_drag_float(
                    tag="inf_noise",
                    default_value=_saved_value(settings, "inf_noise", 0.4),
                    min_value=0.0,
                    max_value=1.0,
                    width=200,
                    format="%.2f",
                    callback=lambda s=None, a=None, u=None: _save_item("inf_noise"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("影响咬字和音质，较为玄学。\n推荐: 0.3-0.5")

        dpg.add_spacer(height=10)

        # 浅扩散 / 增强
        with dpg.collapsing_header(label="浅扩散 / 增强 (官方)", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="浅扩散(改善电音/音质)",
                    tag="inf_shallow",
                    default_value=_saved_value(settings, "inf_shallow", False),
                    callback=lambda s=None, a=None, u=None: _save_item("inf_shallow"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("使用浅扩散模型改善电音问题，与增强器互斥")
                dpg.add_text("扩散步数k_step:")
                dpg.add_drag_int(
                    tag="inf_kstep",
                    default_value=_saved_value(settings, "inf_kstep", 100),
                    min_value=10,
                    max_value=500,
                    width=200,
                    callback=lambda s=None, a=None, u=None: _save_item("inf_kstep"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("扩散步数，越大越接近扩散模型结果。\n推荐: 50-150")
                dpg.add_checkbox(
                    label="二次编码",
                    tag="inf_second_enc",
                    default_value=_saved_value(settings, "inf_second_enc", False),
                    callback=lambda s=None, a=None, u=None: _save_item("inf_second_enc"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("浅扩散前对原始音频二次编码，玄学选项")
            with dpg.group(horizontal=True):
                dpg.add_text("扩散模型:")
                dpg.add_combo([], tag="inf_diff_model", width=400,
                              callback=lambda s=None, a=None, u=None: _save_item("inf_diff_model"))
                dpg.add_button(label="刷新", callback=lambda: _refresh_diff_models(state))
                with dpg.tooltip("inf_diff_model"):
                    dpg.add_text("选择扩散模型，默认使用最新的 model_*.pt")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="NSF-HIFIGAN增强器",
                    tag="inf_enhance",
                    default_value=_saved_value(settings, "inf_enhance", False),
                    callback=lambda s=None, a=None, u=None: _save_item("inf_enhance"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("对训练集少的模型有音质增强效果\n训练好的模型可能反效果，与浅扩散互斥")
                dpg.add_checkbox(
                    label="特征检索",
                    tag="inf_feature_retrieval",
                    default_value=_saved_value(settings, "inf_feature_retrieval", False),
                    callback=lambda s=None, a=None, u=None: _save_item("inf_feature_retrieval"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("使用特征检索，需要训练检索索引")
            dpg.add_text("扩散模型自动从 logs/<工程>/diffusion 加载，或手动选择上方下拉框中的模型。", color=(150, 150, 150))

        dpg.add_spacer(height=10)

        # 聚类/特征检索
        with dpg.collapsing_header(label="聚类/特征检索(可选)"):
            with dpg.group(horizontal=True):
                dpg.add_text("聚类/检索模型:")
                dpg.add_input_text(
                    tag="inf_cluster",
                    width=400,
                    default_value=_saved_value(settings, "inf_cluster", ""),
                    callback=lambda s=None, a=None, u=None: _save_item("inf_cluster"))
                dpg.add_button(label="选择", callback=lambda: dpg.show_item("inf_cluster_dialog"))
            with dpg.group(horizontal=True):
                dpg.add_text("混合比例:")
                dpg.add_drag_float(
                    tag="inf_cluster_ratio",
                    default_value=_saved_value(settings, "inf_cluster_ratio", 0.0),
                    min_value=0.0,
                    max_value=1.0,
                    width=200,
                    format="%.2f",
                    callback=lambda s=None, a=None, u=None: _save_item("inf_cluster_ratio"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("聚类/检索模型混合比例。\n0: 不使用\n0.5-0.8: 推荐范围")

        dpg.add_spacer(height=10)

        # 高级
        with dpg.collapsing_header(label="高级"):
            with dpg.group(horizontal=True):
                dpg.add_text("pad(秒):")
                dpg.add_drag_float(
                    tag="inf_pad",
                    default_value=_saved_value(settings, "inf_pad", 0.5),
                    min_value=0.0,
                    max_value=2.0,
                    width=150,
                    format="%.2f",
                    callback=lambda s=None, a=None, u=None: _save_item("inf_pad"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("推理音频pad秒数，避免开头结尾异响。\n推荐: 0.5")
                dpg.add_text("强制切片(秒,0=自动):")
                dpg.add_drag_float(
                    tag="inf_clip",
                    default_value=_saved_value(settings, "inf_clip", 0.0),
                    min_value=0.0,
                    max_value=30.0,
                    width=150,
                    format="%.1f",
                    callback=lambda s=None, a=None, u=None: _save_item("inf_clip"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("强制切片长度，0为自动。\n长音频可设5-15秒")
                dpg.add_text("响度包络(0-1):")
                dpg.add_drag_float(
                    tag="inf_lea",
                    default_value=_saved_value(settings, "inf_lea", 1.0),
                    min_value=0.0,
                    max_value=1.0,
                    width=150,
                    format="%.2f",
                    callback=lambda s=None, a=None, u=None: _save_item("inf_lea"))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("输入源响度包络替换输出响度包络融合比例。\n1: 完全使用输出\n0: 完全使用输入")

        dpg.add_spacer(height=10)

        # 参数预设
        with dpg.collapsing_header(label="参数预设"):
            with dpg.group(horizontal=True):
                dpg.add_text("预设:")
                dpg.add_combo([], tag="inf_preset", width=200,
                              callback=lambda s=None, a=None, u=None: _save_item("inf_preset"))
                dpg.add_button(label="加载", callback=lambda: _load_preset(state))
                dpg.add_button(label="刷新", callback=lambda: _refresh_presets())
                dpg.add_text("另存为:")
                dpg.add_input_text(tag="inf_preset_name", width=150)
                dpg.add_button(label="保存", callback=lambda: _save_preset())
                dpg.add_button(label="删除", callback=lambda: _delete_preset())
                dpg.add_text("", tag="preset_msg", color=(255, 255, 100))

        dpg.add_spacer(height=10)

        # 开始转换
        dpg.add_button(label="开始转换", callback=lambda: _run_infer(state), width=150)

        dpg.add_spacer(height=10)

        # 播放器
        with dpg.collapsing_header(label="播放器", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="▶ 播放", tag="play_btn", callback=lambda: _play(state), width=80)
                dpg.add_button(label="⏸ 暂停", callback=lambda: _pause(state), width=80)
                dpg.add_button(label="⏹ 停止", callback=lambda: _stop_play(state), width=80)
                dpg.add_text("00:00", tag="play_time")
                dpg.add_slider_float(tag="play_seek", default_value=0, min_value=0, max_value=1000, width=400,
                                     callback=lambda: _seek(state), format="")
                dpg.add_text("00:00", tag="play_dur")
            with dpg.group(horizontal=True):
                dpg.add_text("音量:")
                dpg.add_drag_int(tag="play_vol", default_value=100, min_value=0, max_value=150, width=300,
                                 callback=lambda: _set_volume(state))
                dpg.add_text("100%", tag="play_vol_pct")
                dpg.add_text("", tag="out_path", color=(255, 255, 100))

        dpg.add_spacer(height=10)

        # 日志
        with dpg.group(horizontal=True):
            dpg.add_text("推理日志:")
            dpg.add_button(label="复制全部", callback=lambda: _copy_infer_log(state))
            dpg.add_button(label="清除内容", callback=lambda: clear_job_log(state, "infer", "inf_log", "out_path"))
        with dpg.child_window(tag="inf_log_win", height=230, border=True, horizontal_scrollbar=True):
            dpg.add_text(tag="inf_log", default_value="")

    # 聚类模型文件对话框（保留 DPG 的，因为不常用）
    with dpg.file_dialog(show=False, tag="inf_cluster_dialog",
                         callback=lambda s, d: dpg.set_value("inf_cluster", d["file_path_name"]),
                         width=700, height=400):
        dpg.add_file_extension(".*")
        dpg.add_file_extension(".pt", color=(0, 255, 0))
        dpg.add_file_extension(".pth", color=(0, 255, 0))
        dpg.add_file_extension(".pkl", color=(0, 255, 0))


def _refresh_ckpts(state):
    if not state["current_project"]:
        return
    proj = state["current_project"]
    ckpts = config.list_ckpts(proj)
    labels = [config.ckpt_label(p) for p in ckpts]
    state["ckpt_map"] = {config.ckpt_label(p): p for p in ckpts}
    saved = config.load_settings().get("infer", {})
    ckpt = _select_value(labels, dpg.get_value("inf_ckpt"), saved.get("inf_ckpt", ""))
    dpg.configure_item("inf_ckpt", items=labels)
    dpg.set_value("inf_ckpt", ckpt)
    _save_infer_setting("inf_ckpt", ckpt)
    spks = config.list_speakers(proj)
    spk = _select_value(spks, dpg.get_value("inf_spk"), saved.get("inf_spk", ""))
    dpg.configure_item("inf_spk", items=spks)
    dpg.set_value("inf_spk", spk)
    _save_infer_setting("inf_spk", spk)

    # 同时刷新扩散模型列表
    _refresh_diff_models(state)


def _refresh_diff_models(state):
    """刷新扩散模型列表"""
    if not state["current_project"]:
        return
    proj = state["current_project"]
    diff_ckpts = config.list_diff_ckpts(proj)
    labels = [config.ckpt_label(p) for p in diff_ckpts]
    state["diff_ckpt_map"] = {config.ckpt_label(p): p for p in diff_ckpts}
    saved = config.load_settings().get("infer", {})
    diff_model = _select_value(labels, dpg.get_value("inf_diff_model"), saved.get("inf_diff_model", ""))
    dpg.configure_item("inf_diff_model", items=labels)
    dpg.set_value("inf_diff_model", diff_model)
    _save_infer_setting("inf_diff_model", diff_model)


def _refresh_presets():
    presets = config.list_presets()
    saved = config.load_settings().get("infer", {}).get("inf_preset", "")
    preset = _select_value(presets, dpg.get_value("inf_preset"), saved)
    dpg.configure_item("inf_preset", items=presets)
    dpg.set_value("inf_preset", preset)


def _load_preset(state):
    name = dpg.get_value("inf_preset")
    if not name:
        return
    try:
        data = config.load_preset(name)
        for k, v in data.items():
            if dpg.does_item_exist(k):
                dpg.set_value(k, v)
        _save_all_infer_settings()
        dpg.set_value("preset_msg", f"已加载 {name}")
    except Exception as e:
        dpg.set_value("preset_msg", f"加载失败: {e}")


def _save_preset():
    name = dpg.get_value("inf_preset_name") or dpg.get_value("inf_preset")
    if not name:
        dpg.set_value("preset_msg", "预设名为空")
        return
    keys = ["inf_trans", "inf_slicedb", "inf_f0method", "inf_autof0", "inf_cluster_ratio",
            "inf_noise", "inf_pad", "inf_clip", "inf_device", "inf_spk", "inf_shallow",
            "inf_kstep", "inf_second_enc", "inf_enhance", "inf_feature_retrieval", "inf_lea"]
    data = {k: dpg.get_value(k) for k in keys if dpg.does_item_exist(k)}
    try:
        saved = config.save_preset(name, data)
        _refresh_presets()
        dpg.set_value("inf_preset", saved)
        dpg.set_value("preset_msg", f"已保存 {saved}")
    except Exception as e:
        dpg.set_value("preset_msg", f"保存失败: {e}")


def _delete_preset():
    name = dpg.get_value("inf_preset")
    if not name:
        return
    config.delete_preset(name)
    _refresh_presets()
    dpg.set_value("inf_preset", "")
    dpg.set_value("preset_msg", "已删除")


def _run_infer(state):
    if state["jobs"]["infer"].running():
        return
    if not state["current_project"]:
        dpg.set_value("out_path", "请先选择工程。")
        return
    proj = state["current_project"]
    input_path = dpg.get_value("inf_input")
    if not input_path or not os.path.exists(input_path):
        state["jobs"]["infer"].buf.clear()
        state["jobs"]["infer"].buf.feed("请先选择有效的输入音频。\n")
        return
    ckpt_label = dpg.get_value("inf_ckpt")
    if ckpt_label not in state.get("ckpt_map", {}):
        state["jobs"]["infer"].buf.clear()
        state["jobs"]["infer"].buf.feed("请先选择模型（训练出 G_*.pth 后点刷新）。\n")
        return
    spk = dpg.get_value("inf_spk")
    if not spk:
        state["jobs"]["infer"].buf.clear()
        state["jobs"]["infer"].buf.feed("请先选择说话人。\n")
        return

    model_path = state["ckpt_map"][ckpt_label]
    out_dir = os.path.join(backend.ROOT, "infer_out")
    os.makedirs(out_dir, exist_ok=True)

    # 收集参数
    trans = dpg.get_value("inf_trans")
    slice_db = dpg.get_value("inf_slicedb")
    noise = dpg.get_value("inf_noise")
    pad = dpg.get_value("inf_pad")
    clip = dpg.get_value("inf_clip")
    f0_method = dpg.get_value("inf_f0method")
    device = dpg.get_value("inf_device")
    auto_f0 = dpg.get_value("inf_autof0")
    enhance = dpg.get_value("inf_enhance")
    feature_retrieval = dpg.get_value("inf_feature_retrieval")
    cluster_path = dpg.get_value("inf_cluster")
    cluster_ratio = dpg.get_value("inf_cluster_ratio")
    shallow = dpg.get_value("inf_shallow")
    k_step = dpg.get_value("inf_kstep")
    second_enc = dpg.get_value("inf_second_enc")
    lea = dpg.get_value("inf_lea")

    diff_model = None
    diff_config = None
    if shallow:
        # 优先使用用户选择的扩散模型
        diff_label = dpg.get_value("inf_diff_model")
        if diff_label and diff_label in state.get("diff_ckpt_map", {}):
            diff_model = state["diff_ckpt_map"][diff_label]
            diff_config = config.diff_config_path(proj)
        else:
            # 回退到自动选择最新的
            diff_ckpts = config.list_diff_ckpts(proj)
            if diff_ckpts:
                diff_model = diff_ckpts[0]
                diff_config = config.diff_config_path(proj)

    _save_all_infer_settings()
    cfg_path = config.config_path(proj)
    cmd = backend.infer_cmd(model_path, cfg_path, input_path, out_dir, trans, spk, slice_db,
                            noise, pad, clip, f0_method, device, auto_f0, enhance, feature_retrieval,
                            cluster_path, cluster_ratio, shallow, k_step, second_enc, lea,
                            diff_model, diff_config)

    state["out_dir"] = out_dir
    state["out_before"] = {
        p: os.path.getmtime(p)
        for p in glob.glob(os.path.join(out_dir, "*"))
    }
    dpg.set_value("out_path", "转换中...")
    state["jobs"]["infer"].buf.clear()
    state["jobs"]["infer"].start(cmd)


def _play(state):
    out = state.get("out")
    if out and os.path.exists(out):
        if state.get("loaded") != out:
            state["player"].load(out)
            state["loaded"] = out
        state["player"].set_volume(dpg.get_value("play_vol"))
        state["player"].play()


def _pause(state):
    state["player"].pause()


def _stop_play(state):
    state["player"].stop()


def _seek(state):
    if state["player"].total:
        state["player"].seek(dpg.get_value("play_seek") / 1000.0)


def _set_volume(state):
    vol = dpg.get_value("play_vol")
    state["player"].set_volume(vol)
    dpg.set_value("play_vol_pct", f"{int(vol)}%")


def _browse_input_file(state):
    """使用系统文件对话框选择输入音频"""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 置顶

    file_path = filedialog.askopenfilename(
        title="选择输入音频",
        filetypes=[
            ("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
            ("所有文件", "*.*")
        ]
    )
    root.destroy()

    if file_path:
        # 更新下拉框
        dpg.set_value("inf_input", file_path)
        _save_infer_setting("inf_input", file_path)

        # 保存到历史记录
        settings = config.load_settings()
        recent = settings.get("recent_inputs", [])
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        recent = recent[:10]  # 只保留最近10个
        settings["recent_inputs"] = recent
        config.save_settings(settings)

        # 更新下拉框选项
        dpg.configure_item("inf_input", items=recent, default_value=file_path)


def _copy_infer_log(state):
    """复制推理日志到剪贴板"""
    text = state["jobs"]["infer"].buf.snapshot()
    try:
        import subprocess
        subprocess.run(['clip'], input=text.encode('utf-16le'), check=True)
        dpg.set_value("out_path", "日志已复制到剪贴板")
    except Exception as e:
        dpg.set_value("out_path", f"复制失败: {e}")
