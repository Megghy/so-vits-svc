import json
import sys
import types
import unittest
from unittest import mock

import torch


class FakeEmbedder:
    def __init__(self):
        self.inputs = []

    def embed(self, wav16k):
        self.inputs.append(wav16k)
        return wav16k[:2]


class EvalSpeakerSimilarityTests(unittest.TestCase):
    def test_speaker_similarity_detaches_audio_before_embedding(self):
        from eval_metrics import speaker_similarity

        embedder = FakeEmbedder()
        generated = torch.tensor([1.0, 0.0], requires_grad=True)
        reference = torch.tensor([0.0, 1.0], requires_grad=True)

        score = speaker_similarity(generated, reference, 16000, embedder)

        self.assertAlmostEqual(0.0, score, places=6)
        self.assertFalse(embedder.inputs[0].requires_grad)
        self.assertFalse(embedder.inputs[1].requires_grad)

    def test_collect_speaker_similarity_scalars_logs_mean_and_distance(self):
        from eval_metrics import collect_speaker_similarity_scalars

        class FakeMetric:
            def __init__(self):
                self.scores = [1.0, 0.25]

            def score(self, generated, reference, sample_rate):
                return self.scores.pop(0)

        scalars = collect_speaker_similarity_scalars(
            [
                (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])),
                (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
            ],
            sample_rate=16000,
            metric=FakeMetric(),
        )

        self.assertEqual(
            {
                "eval/speaker_similarity/audio_0": 1.0,
                "eval/speaker_similarity/audio_1": 0.25,
                "eval/speaker_similarity/mean": 0.625,
                "eval/speaker_distance/mean": 0.375,
            },
            scalars,
        )

    def test_speaker_similarity_metric_defaults_to_cpu(self):
        captured = {}

        class FakeSpeakerEmbedder:
            def __init__(self, model_path, device=None):
                captured["model_path"] = model_path
                captured["device"] = device

        fake_module = types.SimpleNamespace(SpeakerEmbedder=FakeSpeakerEmbedder)
        previous_module = sys.modules.get("vencoder.eta_speaker")
        sys.modules["vencoder.eta_speaker"] = fake_module
        try:
            from eval_metrics import SpeakerSimilarityMetric

            SpeakerSimilarityMetric(model_path="fake-speaker-model")
        finally:
            if previous_module is None:
                sys.modules.pop("vencoder.eta_speaker", None)
            else:
                sys.modules["vencoder.eta_speaker"] = previous_module

        self.assertEqual("fake-speaker-model", captured["model_path"])
        self.assertEqual("cpu", captured["device"])

    def test_training_speaker_metric_cache_uses_cpu_even_when_cuda_exists(self):
        import train

        captured = {}

        class FakeMetric:
            def __init__(self, model_path, device=None):
                captured["model_path"] = model_path
                captured["device"] = device

        hps = types.SimpleNamespace(train=types.SimpleNamespace(eval_speaker_model="fake-speaker-model"))
        train._speaker_similarity_metric = None
        train._speaker_similarity_metric_key = None
        with mock.patch.object(train.torch.cuda, "is_available", return_value=True), \
                mock.patch.object(train, "SpeakerSimilarityMetric", FakeMetric):
            train.get_speaker_similarity_metric(hps)

        self.assertEqual("fake-speaker-model", captured["model_path"])
        self.assertEqual("cpu", captured["device"])

    def test_default_configs_enable_eval_speaker_similarity(self):
        for path in ("configs_template/config_template.json", "configs_template/config_tiny_template.json"):
            with self.subTest(path=path):
                with open(path, encoding="utf-8") as f:
                    cfg = json.load(f)

                self.assertTrue(cfg["train"]["eval_speaker_similarity"])
                self.assertEqual(4, cfg["train"]["eval_speaker_similarity_items"])
                self.assertEqual("microsoft/wavlm-base-plus-sv", cfg["train"]["eval_speaker_model"])

    def test_gui_exposes_eval_speaker_similarity_fields(self):
        from local_gui_dpg import config

        field_paths = {field[0] for field in config.CONFIG_FIELDS}
        grouped_paths = {path for _, paths, _ in config.CONFIG_GROUPS for path in paths}

        self.assertIn("train.eval_speaker_similarity", field_paths)
        self.assertIn("train.eval_speaker_similarity_items", field_paths)
        self.assertIn("train.eval_speaker_model", field_paths)
        self.assertIn("train.eval_speaker_similarity", grouped_paths)
        self.assertIn("train.eval_speaker_similarity_items", grouped_paths)
        self.assertIn("train.eval_speaker_model", grouped_paths)


if __name__ == "__main__":
    unittest.main()
