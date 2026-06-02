# -*- coding: utf-8 -*-
"""日志控件通用操作。"""
import dearpygui.dearpygui as dpg


def clear_job_log(state, job_key, log_tag, status_tag=None):
    state["jobs"][job_key].buf.clear()
    if dpg.does_item_exist(log_tag):
        dpg.set_value(log_tag, "")
    if status_tag and dpg.does_item_exist(status_tag):
        dpg.set_value(status_tag, "日志已清空")
