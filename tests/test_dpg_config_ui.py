import os
import json
import tempfile
import unittest
from unittest import mock

import dearpygui.dearpygui as dpg


class DpgConfigUiTests(unittest.TestCase):
    def setUp(self):
        dpg.create_context()

    def tearDown(self):
        dpg.destroy_context()

    def test_every_config_field_has_an_edit_control(self):
        from local_gui_dpg import config, ui_train

        with dpg.window():
            ui_train._create_config_fields()

        missing = [
            path
            for path, *_ in config.CONFIG_FIELDS
            if not dpg.does_item_exist(f"cfg_{path}")
        ]

        self.assertEqual([], missing)

    def test_initial_visibility_update_does_not_auto_save_default_config_values(self):
        from local_gui_dpg import config, ui_train

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("{}")

            ui_train._global_state = {"current_project": "demo"}
            try:
                with mock.patch.object(config, "config_path", return_value=config_path), \
                        mock.patch.object(config, "save_config") as save_config:
                    with dpg.window():
                        ui_train._create_config_fields()

                    save_config.assert_not_called()
            finally:
                ui_train._global_state = None

    def test_refresh_config_fields_shows_bigvgan_groups_after_loading_config(self):
        from local_gui_dpg import config, ui_train

        cfg = {
            "data": {"sampling_rate": 44100},
            "train": {
                "batch_size": 4,
                "learning_rate": 0.0001,
                "bigvgan_strategy": {"mode": "auto_finetune"},
            },
            "model": {
                "speech_encoder": "vec768l12",
                "vocoder_name": "bigvgan-v2",
                "bigvgan_model": "nvidia/bigvgan_v2_44khz_128band_512x",
                "use_cqt_disc": True,
                "use_mrd_disc": True,
                "use_mbd_disc": True,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with mock.patch.object(config, "config_path", return_value=config_path):
                with dpg.window():
                    ui_train._create_config_fields()
                ui_train.refresh_config_fields("demo")

        self.assertTrue(dpg.get_item_configuration("cfg_model.bigvgan_model")["show"])
        self.assertTrue(dpg.get_item_configuration("cfg_model.use_cqt_disc")["show"])

        bigvgan_group_index = next(
            index
            for index, (name, _, _) in enumerate(config.CONFIG_GROUPS)
            if name.startswith("BigVGAN")
        )
        disc_group_index = next(
            index
            for index, (name, _, _) in enumerate(config.CONFIG_GROUPS)
            if name.startswith("判别器增强")
        )
        self.assertTrue(dpg.get_item_configuration(f"cfg_group_{bigvgan_group_index}")["show"])
        self.assertTrue(dpg.get_item_configuration(f"cfg_group_{disc_group_index}")["show"])

    def test_apply_visibility_hides_empty_bigvgan_groups(self):
        from local_gui_dpg import config, ui_train

        with dpg.window():
            ui_train._create_config_fields()

        dpg.set_value("cfg_model.vocoder_name", "nsf-hifigan")
        ui_train._apply_visibility()

        bigvgan_group_index = next(
            index
            for index, (name, _, _) in enumerate(config.CONFIG_GROUPS)
            if name.startswith("BigVGAN")
        )
        disc_group_index = next(
            index
            for index, (name, _, _) in enumerate(config.CONFIG_GROUPS)
            if name.startswith("判别器增强")
        )

        self.assertFalse(dpg.get_item_configuration(f"cfg_group_{bigvgan_group_index}")["show"])
        self.assertFalse(dpg.get_item_configuration(f"cfg_group_{disc_group_index}")["show"])

    def test_log_panel_uses_single_scroll_container_with_auto_follow(self):
        from local_gui_dpg import ui_log

        with dpg.window():
            ui_log.add_log_panel("demo_log", 120)

        self.assertTrue(dpg.does_item_exist("demo_log_win"))
        self.assertTrue(dpg.does_item_exist("demo_log_follow"))
        self.assertTrue(dpg.get_value("demo_log_follow"))

        ui_log.set_log_text("demo_log", "a\nb\nc")

        self.assertEqual("a\nb\nc", dpg.get_value("demo_log"))
        self.assertGreater(dpg.get_item_configuration("demo_log")["height"], 10)


if __name__ == "__main__":
    unittest.main()
