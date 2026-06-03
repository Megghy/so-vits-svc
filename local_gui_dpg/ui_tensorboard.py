# -*- coding: utf-8 -*-
"""TensorBoard 管理标签页 UI"""
import os
import webbrowser
import dearpygui.dearpygui as dpg
from . import config, backend, ui_dialogs
from .ui_log import clear_job_log, add_log_panel


def _browse_tb_logdir():
    p = ui_dialogs.pick_directory("选择 TensorBoard 日志目录")
    if p:
        dpg.set_value("tb_logdir", p)


def create_tensorboard_tab(state):
    """创建 TensorBoard 管理标签页"""
    with dpg.child_window(tag="tab_tensorboard"):
        dpg.add_text("TensorBoard 可视化训练曲线", color=(128, 203, 196))
        dpg.add_spacer(height=10)

        # TensorBoard 控制
        with dpg.collapsing_header(label="TensorBoard 服务", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("日志目录:")
                dpg.add_input_text(tag="tb_logdir", width=500, readonly=True)
                dpg.add_button(label="浏览", callback=lambda: _browse_tb_logdir())

            with dpg.group(horizontal=True):
                dpg.add_text("端口:")
                dpg.add_drag_int(tag="tb_port", default_value=6006, min_value=1024, max_value=65535, width=100)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("TensorBoard 服务端口\n默认: 6006")
                dpg.add_checkbox(label="绑定所有网卡(--bind_all)", tag="tb_bind_all", default_value=False)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("允许局域网访问\n不勾选则只能本机访问")

            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_button(label="启动 TensorBoard", callback=lambda: _start_tensorboard(state), width=150)
                dpg.add_button(label="停止 TensorBoard", callback=lambda: _stop_tensorboard(state), width=150)
                dpg.add_button(label="在浏览器打开", callback=lambda: _open_tensorboard(state), width=150)
                dpg.add_text("", tag="tb_status", color=(255, 255, 100))

            dpg.add_spacer(height=5)
            dpg.add_text("提示：启动后访问 http://localhost:6006", color=(150, 150, 150))

        dpg.add_spacer(height=10)

        # 快捷启动
        with dpg.collapsing_header(label="快捷启动", default_open=True):
            dpg.add_text("为当前工程快速启动 TensorBoard：")
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_button(label="主模型日志", callback=lambda: _quick_start(state, "main"), width=120)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("查看主模型训练曲线\n目录: logs/<工程>/")
                dpg.add_button(label="扩散模型日志", callback=lambda: _quick_start(state, "diff"), width=120)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("查看扩散模型训练曲线\n目录: logs/<工程>/diffusion/")
                dpg.add_button(label="所有日志", callback=lambda: _quick_start(state, "all"), width=120)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("查看所有工程的训练曲线\n目录: logs/")

        dpg.add_spacer(height=10)

        # 日志输出
        dpg.add_text("TensorBoard 日志:")
        with dpg.group(horizontal=True):
            dpg.add_button(label="复制全部", callback=lambda: _copy_tb_log(state))
            dpg.add_button(label="清除内容", callback=lambda: clear_job_log(state, "tb", "tb_log", "tb_status"))
        add_log_panel("tb_log", 400)


def ensure_started(state, logdir):
    """确保 TensorBoard 正在运行；已在运行则跳过。供训练启动时自动调用。
    返回一句状态提示供调用方展示。"""
    port = dpg.get_value("tb_port")
    if state["jobs"]["tb"].running():
        return f"TensorBoard 已在运行(端口 {port})"
    if not os.path.isdir(logdir):
        return "TensorBoard 未启动(日志目录不存在)"
    dpg.set_value("tb_logdir", logdir)
    bind_all = dpg.get_value("tb_bind_all")
    cmd = [backend.PYTHON, "-m", "tensorboard.main",
           "--logdir", logdir,
           "--port", str(port)]
    if bind_all:
        cmd += ["--bind_all"]
    state["jobs"]["tb"].buf.clear()
    state["jobs"]["tb"].start(cmd)
    dpg.set_value("tb_status", f"TensorBoard 已启动在端口 {port}")
    return f"已自动启动 TensorBoard(端口 {port})"


def _start_tensorboard(state):
    """启动 TensorBoard"""
    logdir = dpg.get_value("tb_logdir")
    if not logdir or not os.path.isdir(logdir):
        dpg.set_value("tb_status", "请先选择有效的日志目录")
        return

    if state["jobs"]["tb"].running():
        dpg.set_value("tb_status", "TensorBoard 已在运行")
        return

    port = dpg.get_value("tb_port")
    bind_all = dpg.get_value("tb_bind_all")

    cmd = [backend.PYTHON, "-m", "tensorboard.main",
           "--logdir", logdir,
           "--port", str(port)]
    if bind_all:
        cmd += ["--bind_all"]

    state["jobs"]["tb"].buf.clear()
    state["jobs"]["tb"].start(cmd)
    dpg.set_value("tb_status", f"TensorBoard 已启动在端口 {port}")


def _stop_tensorboard(state):
    """停止 TensorBoard"""
    if not state["jobs"]["tb"].running():
        dpg.set_value("tb_status", "TensorBoard 未运行")
        return

    msg = state["jobs"]["tb"].stop()
    dpg.set_value("tb_status", "TensorBoard 已停止")


def _open_tensorboard(state):
    """在浏览器打开 TensorBoard"""
    port = dpg.get_value("tb_port")
    webbrowser.open(f"http://localhost:{port}")
    dpg.set_value("tb_status", f"已在浏览器打开 http://localhost:{port}")


def _quick_start(state, mode):
    """快捷启动 TensorBoard"""
    if not state["current_project"]:
        dpg.set_value("tb_status", "请先选择工程")
        return

    proj = state["current_project"]
    if mode == "main":
        logdir = config.log_dir(proj)
    elif mode == "diff":
        logdir = config.diff_log_dir(proj)
    else:  # all
        logdir = os.path.join(backend.ROOT, "logs")

    if not os.path.isdir(logdir):
        dpg.set_value("tb_status", f"日志目录不存在: {logdir}")
        return

    dpg.set_value("tb_logdir", logdir)
    _start_tensorboard(state)


def _copy_tb_log(state):
    """复制 TensorBoard 日志到剪贴板"""
    text = state["jobs"]["tb"].buf.snapshot()
    try:
        import subprocess
        subprocess.run(['clip'], input=text.encode('utf-16le'), check=True)
        dpg.set_value("tb_status", "日志已复制到剪贴板")
    except Exception as e:
        dpg.set_value("tb_status", f"复制失败: {e}")
