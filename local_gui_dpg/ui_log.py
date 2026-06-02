# -*- coding: utf-8 -*-
"""日志控件通用操作。"""
import dearpygui.dearpygui as dpg

_FALLBACK_LINE_H = 13.0


def add_log_panel(log_tag, win_tag, height):
    """可选中复制的日志面板：外层 child_window 负责滚动(支持自动滚到底)，
    内层只读多行输入框承载文本(可用鼠标选中、Ctrl+C 复制)。"""
    with dpg.child_window(tag=win_tag, height=height, border=True, horizontal_scrollbar=True):
        dpg.add_input_text(tag=log_tag, multiline=True, readonly=True, width=-1, height=10)


def set_log_text(log_tag, text):
    """更新日志文本，并把输入框高度撑到与内容等高，让外层 child_window 接管滚动。
    (input_text 自身的滚动无法被程序控制，故必须让它不产生内部滚动。)"""
    if not dpg.does_item_exist(log_tag):
        return
    dpg.set_value(log_tag, text)
    size = dpg.get_text_size("Ag")
    line_h = size[1] if size else _FALLBACK_LINE_H
    dpg.configure_item(log_tag, height=int((text.count("\n") + 2) * line_h))


def clear_job_log(state, job_key, log_tag, status_tag=None):
    state["jobs"][job_key].buf.clear()
    set_log_text(log_tag, "")
    if status_tag and dpg.does_item_exist(status_tag):
        dpg.set_value(status_tag, "日志已清空")
