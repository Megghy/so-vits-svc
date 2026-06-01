# -*- coding: utf-8 -*-
"""DPG 主题配置"""
import dearpygui.dearpygui as dpg


def setup_theme():
    """深色现代主题，圆角按钮，蓝色强调色"""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (20, 20, 25))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (25, 25, 30))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 48))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (50, 50, 60))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (60, 60, 72))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (25, 25, 30))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (30, 30, 38))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 120, 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 140, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (50, 100, 180))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (70, 140, 220))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (90, 160, 240))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 50, 60))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (60, 60, 72))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (70, 70, 84))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (40, 40, 48))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (60, 60, 72))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (50, 50, 60))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 230))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (120, 120, 130))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 70))
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, (79, 195, 247))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (79, 195, 247))

            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 4)

    dpg.bind_theme(global_theme)


def setup_font():
    """加载中文字体"""
    import os
    # 缓存字体路径，避免每次启动都检查
    fonts = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhl.ttc",
             r"C:\Windows\Fonts\simhei.ttf"]
    font_path = next((f for f in fonts if os.path.exists(f)), None)
    if font_path:
        try:
            with dpg.font_registry():
                default_font = dpg.add_font(font_path, 16)
                dpg.bind_font(default_font)
        except Exception as e:
            print(f"字体加载失败: {e}")
