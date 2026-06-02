# -*- coding: utf-8 -*-
"""原生文件/目录选择对话框(tkinter)，替代 DPG 内置的 file_dialog。"""


def _hidden_root():
    import tkinter as tk
    r = tk.Tk()
    r.withdraw()
    r.attributes("-topmost", True)
    return r


def pick_directory(title="选择目录", initialdir=None):
    """弹出系统原生目录选择框，返回所选路径；取消返回 None。"""
    from tkinter import filedialog
    r = _hidden_root()
    try:
        path = filedialog.askdirectory(title=title, initialdir=initialdir or None, mustexist=True)
    finally:
        r.destroy()
    return path or None


def pick_file(title="选择文件", filetypes=None, initialdir=None):
    """弹出系统原生文件选择框，返回所选路径；取消返回 None。
    filetypes 形如 [("音频文件", "*.wav *.mp3"), ("所有文件", "*.*")]。"""
    from tkinter import filedialog
    r = _hidden_root()
    try:
        path = filedialog.askopenfilename(
            title=title, filetypes=filetypes or [("所有文件", "*.*")],
            initialdir=initialdir or None)
    finally:
        r.destroy()
    return path or None
