# -*- coding: utf-8 -*-
"""主入口：创建 DPG 窗口，组装所有标签页，启动主循环"""
import os
import time
import glob
import dearpygui.dearpygui as dpg
from . import config, backend, theme
from . import ui_log
from .ui_dataset import create_dataset_tab
from .ui_train import create_train_tab, refresh_config_fields, _refresh_chart, _plot_chart
from .ui_infer import create_infer_tab, _refresh_ckpts, _refresh_presets
from .ui_tensorboard import create_tensorboard_tab


def _fmt_time(sec):
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def _updated_audio_files(out_dir, before):
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


def create_main_window():
    """创建主窗口和全局状态"""
    # 全局状态
    state = {
        "current_project": None,
        "jobs": {k: backend.Job() for k in ("ds", "train", "diff", "tb", "infer", "cluster")},
        "player": backend.Player(),
        "ckpt_map": {},
        "diff_ckpt_map": {},
        "out": None,
        "out_dir": None,
        "out_before": set(),
        "loaded": None,
        "chart_scalars": {},
        "chart_dirty": False,
        "chart_auto_on": False,
        "target_step": 0,
        "patch_after_ds": False,
        "pending_log_scroll": set(),
    }
    # TensorBoard 现在有独立日志窗口，不再并入训练日志

    # 加载设置
    settings = config.load_settings()
    state["current_project"] = settings.get("project")
    if state["current_project"] and state["current_project"] not in config.list_projects():
        state["current_project"] = None

    with dpg.window(tag="main_window", label="so-vits-svc 4.1 一体化工具 (DearPyGui)", width=1400, height=900):
        # 顶部工程选择条
        with dpg.group(horizontal=True):
            dpg.add_text("工程:")
            dpg.add_combo(config.list_projects(), tag="proj_combo", width=250,
                          default_value=state["current_project"] or "",
                          callback=lambda: _switch_project(state))
            dpg.add_button(label="刷新", callback=lambda: _refresh_projects(state))
            dpg.add_text("新建:")
            dpg.add_input_text(tag="proj_new", width=200)
            dpg.add_button(label="创建工程", callback=lambda: _create_project(state))
            dpg.add_text("", tag="proj_msg", color=(255, 255, 100))

        dpg.add_separator()
        dpg.add_spacer(height=5)

        # 标签页
        with dpg.tab_bar():
            with dpg.tab(label="数据集预处理"):
                create_dataset_tab(state)
            with dpg.tab(label="训练"):
                create_train_tab(state)
            with dpg.tab(label="推理"):
                create_infer_tab(state)
            with dpg.tab(label="TensorBoard"):
                create_tensorboard_tab(state)

    return state


def _refresh_projects(state):
    projs = config.list_projects()
    dpg.configure_item("proj_combo", items=projs)


def _switch_project(state):
    proj = dpg.get_value("proj_combo")
    if proj:
        state["current_project"] = proj
        _refresh_project_ui(state)
        dpg.set_value("proj_msg", f"已切换到工程 {proj}")


def _create_project(state):
    name = dpg.get_value("proj_new")
    try:
        name = config.create_project(name)
    except ValueError as e:
        dpg.set_value("proj_msg", str(e))
        return
    state["current_project"] = name
    _refresh_projects(state)
    dpg.set_value("proj_combo", name)
    dpg.set_value("proj_new", "")
    _refresh_project_ui(state)
    dpg.set_value("proj_msg", f"已创建并切换到工程 {name}")


def _refresh_project_ui(state):
    """切换工程后刷新所有相关 UI"""
    if not state["current_project"]:
        return
    proj = state["current_project"]

    # 刷新推理页的模型和说话人列表
    _refresh_ckpts(state)
    _refresh_presets()

    # 刷新训练页的配置字段
    if os.path.exists(config.config_path(proj)):
        refresh_config_fields(proj)

    # 刷新训练曲线
    _refresh_chart(state)


def _update_logs(state):
    """定时刷新所有日志窗口"""
    pending = state.setdefault("pending_log_scroll", set())
    for key, log_tag in [("ds", "ds_log"),
                         ("train", "train_log"),
                         ("diff", "diff_log"),
                         ("infer", "inf_log"),
                         ("cluster", "cluster_log"),
                         ("tb", "tb_log")]:
        if key not in state["jobs"]:
            continue
        job = state["jobs"][key]
        if job.buf.dirty:
            text = job.buf.snapshot()
            if dpg.does_item_exist(log_tag):
                ui_log.set_log_text(log_tag, text)
                if ui_log.log_should_follow(log_tag):
                    pending.add(log_tag)


def _scroll_pending_logs(state):
    """日志文本更新后等一帧布局完成，再把开启自动跟随的日志滚到底。"""
    pending = state.setdefault("pending_log_scroll", set())
    if not pending:
        return
    for log_tag in list(pending):
        if ui_log.log_should_follow(log_tag):
            ui_log.scroll_log_to_bottom(log_tag)
        pending.discard(log_tag)


def _update_player(state):
    """定时刷新播放器进度条"""
    player = state["player"]
    if player.data is None:
        return
    dpg.set_value("play_dur", _fmt_time(player.duration))
    dpg.set_value("play_time", _fmt_time(player.cur_time))
    if player.playing() and player.total:
        dpg.set_value("play_seek", int(player.cur_time / player.duration * 1000))


def _check_jobs(state):
    """检查后台任务完成状态"""
    # 数据集预处理完成
    while not state["jobs"]["ds"].done_q.empty():
        rc = state["jobs"]["ds"].done_q.get()
        if rc == 0 and state.get("patch_after_ds"):
            config.patch_configs_for_project(state["current_project"])
            state["patch_after_ds"] = False
            _refresh_project_ui(state)
        dpg.set_value("ds_pipe_msg",
                      "完成。可去训练页开始训练。" if rc == 0 else f"中止/失败(退出码 {rc})，请看日志。")

    # 训练/扩散/TB 完成（清空队列即可）
    for k in ("train", "diff", "tb"):
        while not state["jobs"][k].done_q.empty():
            state["jobs"][k].done_q.get()

    # 聚类/检索训练完成
    if "cluster" in state["jobs"]:
        while not state["jobs"]["cluster"].done_q.empty():
            rc = state["jobs"]["cluster"].done_q.get()
            if rc == 0:
                dpg.set_value("cluster_msg", "训练完成！模型已保存到 logs/<工程>/ 目录。")
            else:
                dpg.set_value("cluster_msg", f"训练失败(退出码 {rc})，请看日志。")

    # 推理完成
    if not state["jobs"]["infer"].done_q.empty():
        rc = state["jobs"]["infer"].done_q.get()
        audio = _updated_audio_files(state.get("out_dir", ""), state.get("out_before", {}))
        out = max(audio, key=os.path.getmtime) if audio else None
        if rc == 0 and out and os.path.exists(out):
            state["out"] = out
            dpg.set_value("out_path", f"完成: {out}")
            state["player"].load(out)
            state["loaded"] = out
            state["player"].set_volume(dpg.get_value("play_vol"))
            state["player"].play()
        else:
            dpg.set_value("out_path", "转换失败，看日志。")
        # 推理后刷新模型列表
        _refresh_ckpts(state)


def _chart_worker(state):
    """后台线程：训练中定时全量解析 tfevents（重 IO），结果写入 state。
    放后台是为了不阻塞主渲染线程——这步是训练时 UI 卡顿的根因。
    仅在「自动刷新开启」或「设了目标step」时才解析，避免无谓 IO。"""
    while dpg.is_dearpygui_running():
        try:
            if (state["jobs"]["train"].running() and state.get("current_project")
                    and (state.get("chart_auto_on") or state.get("target_step"))):
                state["chart_scalars"] = config.read_scalars(config.log_dir(state["current_project"]))
                state["chart_dirty"] = True
        except Exception:
            pass
        time.sleep(5)


def _auto_refresh_chart(state):
    """主线程消费后台读好的曲线数据：绘图 + 目标step自动停。本身不做重 IO。"""
    if not state.get("chart_dirty"):
        return
    state["chart_dirty"] = False
    scalars = state.get("chart_scalars", {})

    if dpg.get_value("chart_auto"):
        tags = list(scalars.keys())
        cur = dpg.get_value("chart_tag")
        if not cur or cur not in tags:
            cur = "loss/g/total" if "loss/g/total" in tags else (tags[0] if tags else "")
        if cur and cur != dpg.get_value("chart_tag"):
            dpg.set_value("chart_tag", cur)
        if cur:
            _plot_chart(state)

    # 目标step自动停（不受自动刷新开关影响）
    target = state.get("target_step", 0)
    if target:
        cur_step = max((s[-1] for s, _ in scalars.values() if s), default=0)
        if cur_step >= target:
            state["target_step"] = 0
            state["jobs"]["train"].stop()
            dpg.set_value("train_msg", f"已达 {cur_step} step（目标 {target}），自动停止训练。")


def main_loop(state):
    """主循环：定时刷新日志、播放器、任务状态、训练曲线"""
    while dpg.is_dearpygui_running():
        state["chart_auto_on"] = dpg.get_value("chart_auto")
        _update_logs(state)
        _update_player(state)
        _check_jobs(state)
        _auto_refresh_chart(state)
        dpg.render_dearpygui_frame()
        _scroll_pending_logs(state)


def main():
    """主入口"""
    print("正在启动 GUI...")
    print("1/5 创建上下文...")
    dpg.create_context()

    print("2/5 加载字体...")
    theme.setup_font()

    print("3/5 应用主题...")
    theme.setup_theme()

    print("4/5 构建界面...")
    state = create_main_window()

    print("5/5 初始化窗口...")
    dpg.create_viewport(title="so-vits-svc 4.1 一体化工具", width=1400, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

    print("✓ GUI 启动完成")

    # 初始化：刷新工程UI（快速）
    if state["current_project"]:
        _refresh_project_ui(state)

    # 数据集扫描改为异步（避免启动卡顿）
    def _async_scan():
        import time
        time.sleep(0.5)  # 等待窗口显示
        from .ui_dataset import _scan_dataset
        _scan_dataset(state)

    import threading
    threading.Thread(target=_async_scan, daemon=True).start()
    # 训练曲线解析放后台线程，避免阻塞主渲染线程
    threading.Thread(target=_chart_worker, args=(state,), daemon=True).start()

    # 启动主循环
    try:
        main_loop(state)
    finally:
        # 保存设置
        settings = config.load_settings()
        settings.update({
            "volume": int(dpg.get_value("play_vol")),
            "project": state["current_project"],
            "device": dpg.get_value("inf_device"),
            "chart_auto": dpg.get_value("chart_auto"),
            "chart_log": dpg.get_value("chart_log"),
        })
        config.save_settings(settings)

        # 停止所有后台任务（强制结束）
        state["player"].stop()
        for j in state["jobs"].values():
            if j.running():
                try:
                    import platform
                    if platform.system() == "Windows":
                        import subprocess
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(j.proc.pid)],
                                       capture_output=True, timeout=3)
                    else:
                        j.proc.terminate()
                        j.proc.wait(timeout=3)
                except Exception:
                    try:
                        j.proc.kill()
                    except Exception:
                        pass

        dpg.destroy_context()


if __name__ == "__main__":
    main()
