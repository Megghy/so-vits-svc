# -*- coding: utf-8 -*-
"""训练标签页 UI"""
import os
import webbrowser
import dearpygui.dearpygui as dpg
from . import config, backend
from .ui_log import clear_job_log

# 全局 state，供自动保存回调使用
_global_state = None


def create_train_tab(state):
    """创建训练标签页"""
    # 保存 state 到全局，供自动保存回调使用
    global _global_state
    _global_state = state

    with dpg.child_window(tag="tab_train"):
        # 主模型训练
        with dpg.collapsing_header(label="主模型训练 (train.py)", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("到达step自动停(0=不限):")
                dpg.add_input_int(tag="tr_target", default_value=0, width=100, step=0)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("训练到指定step自动停止，0表示不限制")
                dpg.add_button(label="开始训练", callback=lambda: _start_train(state))
                dpg.add_button(label="停止", callback=lambda: _stop_train(state))
                dpg.add_text("", tag="train_msg", color=(255, 255, 100))
            dpg.add_text("自动从 logs/<工程> 最新 G/D_*.pth 续训；需手动放底模 G_0.pth/D_0.pth。", color=(150, 150, 150))
            dpg.add_text("提示：TensorBoard 可在独立标签页启动和管理。", color=(128, 203, 196))

        dpg.add_spacer(height=10)

        # 浅扩散训练
        with dpg.collapsing_header(label="浅扩散训练 (train_diff.py，可选)", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="开始扩散训练", callback=lambda: _start_diff_train(state))
                dpg.add_button(label="停止", callback=lambda: _stop_diff_train(state))
                dpg.add_text("", tag="diff_msg", color=(255, 255, 100))
            dpg.add_text("需先在特征提取勾选 --use_diff。扩散模型存到 logs/<工程>/diffusion。", color=(150, 150, 150))
            dpg.add_text("底模 model_0.pt 放在 pretrain/diffusion/ 下，训练时自动复制。", color=(128, 203, 196))

        dpg.add_spacer(height=10)

        # 聚类/特征检索训练
        with dpg.collapsing_header(label="聚类/特征检索训练 (可选)", default_open=False):
            dpg.add_text("用于推理时改善音色相似度，需先完成特征提取。", color=(150, 150, 150))
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                dpg.add_text("聚类中心数:")
                dpg.add_drag_int(tag="cluster_n", default_value=10000, min_value=1000, max_value=50000, width=150)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("KMeans 聚类中心数量。\n推荐: 10000（默认）\n数据集大可用 20000")
                dpg.add_checkbox(label="使用 GPU 加速", tag="cluster_gpu", default_value=False)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("GPU 加速聚类训练（需要 faiss-gpu）\n小数据集用 CPU 即可")
                dpg.add_button(label="训练聚类模型", callback=lambda: _start_cluster_train(state))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("训练 KMeans 聚类模型\n输出: logs/<工程>/kmeans_10000.pt")

            with dpg.group(horizontal=True):
                dpg.add_button(label="训练特征检索", callback=lambda: _start_index_train(state))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("训练特征检索索引（基于 faiss）\n输出: logs/<工程>/feature_and_index.pkl")
                dpg.add_button(label="停止", callback=lambda: _stop_cluster_train(state))
                dpg.add_text("", tag="cluster_msg", color=(255, 255, 100))

            dpg.add_text("聚类和特征检索二选一即可，推荐特征检索（效果更好）。", color=(128, 203, 196))

        dpg.add_spacer(height=10)

        # 训练参数配置
        with dpg.collapsing_header(label="训练参数 (config.json)", default_open=True):
            _create_config_fields()
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="重新加载", callback=lambda: _reload_config(state))
                dpg.add_button(label="保存到 config.json", callback=lambda: _save_config(state))
                dpg.add_button(label="检查配置", callback=lambda: _check_config(state))
                dpg.add_text("4080S 建议 fp16_run=false + half_type=bf16，开 all_in_mem 加速。", color=(128, 203, 196))
                dpg.add_text("", tag="cfg_msg", color=(255, 255, 100))
            dpg.add_spacer(height=4)
            with dpg.child_window(tag="cfg_check_win", height=130, border=True):
                dpg.add_text("修改参数会自动体检，或点「检查配置」手动运行。",
                             tag="cfg_check_result", wrap=900, color=(150, 150, 150))

        dpg.add_spacer(height=10)

        # 标签页：训练日志 / 扩散日志 / 训练曲线
        with dpg.tab_bar():
            with dpg.tab(label="训练日志"):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="复制全部", callback=lambda: _copy_train_log(state, "train"))
                    dpg.add_button(label="清除内容",
                                   callback=lambda: clear_job_log(state, "train", "train_log", "train_msg"))
                with dpg.child_window(tag="train_log_win", height=380, border=True, horizontal_scrollbar=True):
                    dpg.add_input_text(tag="train_log", multiline=True, readonly=True, width=-1, height=18)

            with dpg.tab(label="扩散日志"):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="复制全部", callback=lambda: _copy_train_log(state, "diff"))
                    dpg.add_button(label="清除内容",
                                   callback=lambda: clear_job_log(state, "diff", "diff_log", "diff_msg"))
                with dpg.child_window(tag="diff_log_win", height=380, border=True, horizontal_scrollbar=True):
                    dpg.add_input_text(tag="diff_log", multiline=True, readonly=True, width=-1, height=18)

            with dpg.tab(label="聚类/检索日志"):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="复制全部", callback=lambda: _copy_train_log(state, "cluster"))
                    dpg.add_button(label="清除内容",
                                   callback=lambda: clear_job_log(state, "cluster", "cluster_log", "cluster_msg"))
                with dpg.child_window(tag="cluster_log_win", height=380, border=True, horizontal_scrollbar=True):
                    dpg.add_input_text(tag="cluster_log", multiline=True, readonly=True, width=-1, height=18)

            with dpg.tab(label="训练曲线"):
                with dpg.group(horizontal=True):
                    dpg.add_text("曲线:")
                    dpg.add_combo([], tag="chart_tag", width=300, callback=lambda: _plot_chart(state))
                    dpg.add_checkbox(label="对数纵轴", tag="chart_log", callback=lambda: _plot_chart(state))
                    dpg.add_checkbox(label="自动刷新(5s)", tag="chart_auto", default_value=False)
                    dpg.add_button(label="刷新曲线", callback=lambda: _refresh_chart(state))
                dpg.add_spacer(height=5)
                dpg.add_text("提示：完整的 TensorBoard 功能请前往 TensorBoard 标签页。", color=(128, 203, 196))
                dpg.add_spacer(height=5)
                with dpg.plot(tag="chart_plot", label="", height=400, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="step", tag="chart_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="value", tag="chart_y")
                dpg.add_text("", tag="chart_msg", color=(255, 255, 100))


def _add_config_field(path):
    """渲染单个配置字段(标签+控件+tooltip)。"""
    field = config.CONFIG_FIELD_MAP.get(path)
    if field is None:
        dpg.add_text("")
        dpg.add_text("")
        return
    _, name, ftype, default, range_opts, tooltip = field
    tag = f"cfg_{path}"
    dpg.add_text(name + ":")
    if ftype == "bool":
        dpg.add_checkbox(tag=tag, default_value=default, callback=_auto_save_config)
    elif ftype == "combo":
        dpg.add_combo(range_opts, tag=tag, default_value=default, width=150, callback=_auto_save_config)
    elif ftype == "int":
        if range_opts:
            dpg.add_drag_int(tag=tag, default_value=default, min_value=range_opts[0],
                             max_value=range_opts[1], width=150, callback=_auto_save_config)
        else:
            dpg.add_input_int(tag=tag, default_value=default, width=150, step=0, callback=_auto_save_config)
    elif ftype == "float":
        if range_opts:
            dpg.add_drag_float(tag=tag, default_value=default, min_value=range_opts[0],
                               max_value=range_opts[1], width=150, format="%.6f", callback=_auto_save_config)
        else:
            dpg.add_input_float(tag=tag, default_value=default, width=150, format="%.6f", step=0, callback=_auto_save_config)
    if tooltip:
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text(tooltip)


def _create_config_fields():
    """按分组渲染配置字段，每组一个折叠面板，组内 2 列网格。"""
    for group_name, paths, default_open in config.CONFIG_GROUPS:
        with dpg.collapsing_header(label=group_name, default_open=default_open):
            with dpg.table(header_row=False, borders_innerH=True, borders_outerH=True,
                           borders_innerV=True, borders_outerV=True):
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                dpg.add_table_column()
                for i in range(0, len(paths), 2):
                    with dpg.table_row():
                        for j in range(2):
                            idx = i + j
                            if idx >= len(paths):
                                dpg.add_text("")
                                dpg.add_text("")
                                continue
                            _add_config_field(paths[idx])


def _start_train(state):
    if not state["current_project"]:
        dpg.set_value("train_msg", "请先创建/选择工程。")
        return
    proj = state["current_project"]
    if not os.path.exists(config.config_path(proj)):
        dpg.set_value("train_msg", "配置不存在，先完成预处理。")
        return
    if state["jobs"]["train"].running():
        dpg.set_value("train_msg", "训练已在运行。")
        return

    target = dpg.get_value("tr_target")
    state["target_step"] = target if target > 0 else 0
    step = config.latest_step(proj)
    tip = f"从 step {step} 续训..." if step else "从底模/零开始训练..."
    if state["target_step"]:
        tip += f" 到 {state['target_step']} step 自动停。"
    dpg.set_value("train_msg", tip)

    cmd = backend.train_cmd(config.config_path(proj), proj)
    state["jobs"]["train"].start(cmd)


def _stop_train(state):
    state["target_step"] = 0
    msg = state["jobs"]["train"].stop()
    dpg.set_value("train_msg", msg)


def _start_diff_train(state):
    if not state["current_project"]:
        dpg.set_value("diff_msg", "请先创建/选择工程。")
        return
    proj = state["current_project"]
    if not os.path.exists(config.diff_config_path(proj)):
        dpg.set_value("diff_msg", "扩散配置不存在，先跑预处理(勾选--use_diff)。")
        return

    # 自动复制底模
    diff_dir = config.diff_log_dir(proj)
    os.makedirs(diff_dir, exist_ok=True)
    target_base = os.path.join(diff_dir, "model_0.pt")

    if not os.path.exists(target_base):
        source_base = os.path.join(config.ROOT, "pretrain", "diffusion", "model_0.pt")
        if os.path.exists(source_base):
            import shutil
            shutil.copy2(source_base, target_base)
            dpg.set_value("diff_msg", "已自动复制底模 model_0.pt，启动训练...")
        else:
            dpg.set_value("diff_msg", "警告：未找到底模 pretrain/diffusion/model_0.pt，将从零开始训练...")

    cmd = backend.diff_train_cmd(config.diff_config_path(proj))
    state["jobs"]["diff"].start(cmd)
    if os.path.exists(target_base):
        dpg.set_value("diff_msg", "扩散训练启动...")
    else:
        dpg.set_value("diff_msg", "扩散训练启动(从零开始)...")


def _stop_diff_train(state):
    msg = state["jobs"]["diff"].stop()
    dpg.set_value("diff_msg", msg)


def _reload_config(state):
    if not state["current_project"]:
        dpg.set_value("cfg_msg", "请先选择工程。")
        return
    proj = state["current_project"]
    if not os.path.exists(config.config_path(proj)):
        dpg.set_value("cfg_msg", "配置不存在，先跑预处理。")
        return
    cfg = config.load_config(proj)
    for path, _, ftype, _, _, _ in config.CONFIG_FIELDS:
        tag = f"cfg_{path}"
        try:
            keys = path.split(".")
            node = cfg
            for k in keys:
                node = node[k]
            dpg.set_value(tag, node)
        except KeyError:
            pass
    dpg.set_value("cfg_msg", "已从配置文件重新加载。")
    _check_config(state, silent=True)


def _save_config(state):
    if not state["current_project"]:
        dpg.set_value("cfg_msg", "请先选择工程。")
        return
    proj = state["current_project"]
    if not os.path.exists(config.config_path(proj)):
        dpg.set_value("cfg_msg", "配置不存在，先跑预处理。")
        return
    values = {}
    for path, _, ftype, _, _, _ in config.CONFIG_FIELDS:
        tag = f"cfg_{path}"
        values[path] = dpg.get_value(tag)
    try:
        changed = config.save_config(proj, values)
        dpg.set_value("cfg_msg", f"已保存 {len(changed)} 项改动。" if changed else "无改动。")
        _check_config(state, silent=True)
    except Exception as e:
        dpg.set_value("cfg_msg", f"保存失败：{e}")


def _check_config(state, silent=False):
    """配置静态体检，结果写入 cfg_check_result 并按严重度着色。silent=True 时不抢占 cfg_msg。"""
    if not state or not state.get("current_project"):
        if not silent:
            dpg.set_value("cfg_msg", "请先选择工程。")
        return
    proj = state["current_project"]
    if not os.path.exists(config.config_path(proj)):
        if not silent:
            dpg.set_value("cfg_msg", "配置不存在，先跑预处理。")
        return
    issues = config.check_config(config.load_config(proj))
    icon = {"error": "[错误] ", "warn": "[警告] ", "info": "[提示] ", "ok": "[OK] "}
    dpg.set_value("cfg_check_result", "\n".join(icon.get(lv, "") + msg for lv, msg in issues))
    n_err = sum(1 for lv, _ in issues if lv == "error")
    n_warn = sum(1 for lv, _ in issues if lv == "warn")
    color = (255, 120, 120) if n_err else (255, 215, 120) if n_warn else (150, 220, 150)
    dpg.configure_item("cfg_check_result", color=color)
    if not silent:
        summ = "未发现问题。" if not (n_err or n_warn) else f"{n_err} 个错误、{n_warn} 个警告。"
        dpg.set_value("cfg_msg", "已检查：" + summ)


def _refresh_chart(state):
    if not state["current_project"]:
        return
    proj = state["current_project"]
    scalars = config.read_scalars(config.log_dir(proj))
    state["chart_scalars"] = scalars
    tags = list(scalars.keys())
    cur = dpg.get_value("chart_tag")
    if cur not in tags:
        cur = "loss/g/total" if "loss/g/total" in tags else (tags[0] if tags else "")
    dpg.configure_item("chart_tag", items=tags, default_value=cur)
    if cur:
        _plot_chart(state)


def _plot_chart(state):
    tag = dpg.get_value("chart_tag")
    if not tag or tag not in state.get("chart_scalars", {}):
        return
    steps, vals = state["chart_scalars"][tag]
    if not steps:
        return

    # 清空旧数据
    dpg.delete_item("chart_y", children_only=True)
    dpg.delete_item("chart_x", children_only=True)

    # 重新创建轴
    log_scale = dpg.get_value("chart_log")
    dpg.set_axis_limits("chart_x", steps[0], steps[-1])
    if log_scale:
        dpg.configure_item("chart_y", log_scale=True)
        dpg.set_axis_limits_auto("chart_y")
    else:
        dpg.configure_item("chart_y", log_scale=False)
        dpg.set_axis_limits_auto("chart_y")

    # 添加线条
    dpg.add_line_series(steps, vals, label=tag, parent="chart_y")
    dpg.set_value("chart_msg", f"step {steps[-1]}  =  {vals[-1]:.5f}")


def _copy_train_log(state, job_key):
    """复制训练日志到剪贴板"""
    status_tag = {"train": "train_msg", "diff": "diff_msg", "cluster": "cluster_msg"}.get(job_key, "train_msg")
    text = state["jobs"][job_key].buf.snapshot()
    try:
        import subprocess
        subprocess.run(['clip'], input=text.encode('utf-16le'), check=True)
        dpg.set_value(status_tag, "日志已复制")
    except Exception as e:
        dpg.set_value(status_tag, f"复制失败: {e}")


def _auto_save_config():
    """配置字段修改时自动保存"""
    if not _global_state or not _global_state.get("current_project"):
        return
    proj = _global_state["current_project"]
    if not os.path.exists(config.config_path(proj)):
        return

    values = {}
    for path, _, ftype, _, _, _ in config.CONFIG_FIELDS:
        tag = f"cfg_{path}"
        if dpg.does_item_exist(tag):
            values[path] = dpg.get_value(tag)

    try:
        changed = config.save_config(proj, values)
        if changed:
            dpg.set_value("cfg_msg", f"已自动保存 {len(changed)} 项")
        _check_config(_global_state, silent=True)
    except Exception as e:
        dpg.set_value("cfg_msg", f"自动保存失败: {e}")


def _start_cluster_train(state):
    """训练聚类模型"""
    if not state["current_project"]:
        dpg.set_value("cluster_msg", "请先选择工程。")
        return
    proj = state["current_project"]
    if not os.path.exists(config.config_path(proj)):
        dpg.set_value("cluster_msg", "配置不存在，先完成预处理。")
        return

    # 检查是否有 .soft.pt 特征文件
    dataset_dir = backend.DATASET_44K
    has_features = False
    for spk_dir in os.listdir(dataset_dir):
        spk_path = os.path.join(dataset_dir, spk_dir)
        if os.path.isdir(spk_path):
            if any(f.endswith(".soft.pt") for f in os.listdir(spk_path)):
                has_features = True
                break

    if not has_features:
        dpg.set_value("cluster_msg", "未找到特征文件(.soft.pt)，请先运行特征提取。")
        return

    n_clusters = dpg.get_value("cluster_n")
    use_gpu = dpg.get_value("cluster_gpu")
    output_dir = config.log_dir(proj)

    cmd = backend.cluster_train_cmd(dataset_dir, output_dir, n_clusters, use_gpu)
    state["jobs"]["cluster"] = state["jobs"].get("cluster") or backend.Job()
    state["jobs"]["cluster"].buf.clear()
    state["jobs"]["cluster"].start(cmd)
    dpg.set_value("cluster_msg", f"开始训练聚类模型（{n_clusters} 个中心）...")


def _start_index_train(state):
    """训练特征检索索引"""
    if not state["current_project"]:
        dpg.set_value("cluster_msg", "请先选择工程。")
        return
    proj = state["current_project"]
    cfg_path = config.config_path(proj)
    if not os.path.exists(cfg_path):
        dpg.set_value("cluster_msg", "配置不存在，先完成预处理。")
        return

    dataset_dir = backend.DATASET_44K
    output_dir = config.log_dir(proj)

    cmd = backend.index_train_cmd(dataset_dir, cfg_path, output_dir)
    state["jobs"]["cluster"] = state["jobs"].get("cluster") or backend.Job()
    state["jobs"]["cluster"].buf.clear()
    state["jobs"]["cluster"].start(cmd)
    dpg.set_value("cluster_msg", "开始训练特征检索索引...")


def _stop_cluster_train(state):
    """停止聚类/检索训练"""
    if "cluster" in state["jobs"]:
        msg = state["jobs"]["cluster"].stop()
        dpg.set_value("cluster_msg", msg)
    else:
        dpg.set_value("cluster_msg", "没有正在运行的训练。")
