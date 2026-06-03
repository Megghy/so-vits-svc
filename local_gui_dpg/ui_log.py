# -*- coding: utf-8 -*-
"""日志控件通用操作。"""
import dearpygui.dearpygui as dpg

_FALLBACK_LINE_H = 13.0


def log_window_tag(log_tag):
    return f"{log_tag}_win"


def log_follow_tag(log_tag):
    return f"{log_tag}_follow"


def add_log_panel(log_tag, height):
    """固定高度日志区：外层唯一滚动容器，内层只读文本框用于选择复制。"""
    dpg.add_checkbox(label="自动跟随", tag=log_follow_tag(log_tag), default_value=True)
    with dpg.child_window(tag=log_window_tag(log_tag), height=height, border=True,
                          horizontal_scrollbar=True):
        dpg.add_input_text(tag=log_tag, multiline=True, readonly=True, width=-1, height=10)


def set_log_text(log_tag, text):
    if not dpg.does_item_exist(log_tag):
        return
    dpg.set_value(log_tag, text)
    size = dpg.get_text_size("Ag")
    line_h = size[1] if size else _FALLBACK_LINE_H
    dpg.configure_item(log_tag, height=max(10, int((text.count("\n") + 3) * line_h)))


def log_should_follow(log_tag):
    tag = log_follow_tag(log_tag)
    return not dpg.does_item_exist(tag) or bool(dpg.get_value(tag))


def scroll_log_to_bottom(log_tag):
    tag = log_window_tag(log_tag)
    if dpg.does_item_exist(tag):
        dpg.set_y_scroll(tag, dpg.get_y_scroll_max(tag))


def clear_job_log(state, job_key, log_tag, status_tag=None):
    state["jobs"][job_key].buf.clear()
    set_log_text(log_tag, "")
    if status_tag and dpg.does_item_exist(status_tag):
        dpg.set_value(status_tag, "日志已清空")
