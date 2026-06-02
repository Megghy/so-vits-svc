import sys
import types
import unittest

import torch

sys.modules.setdefault("faiss", types.SimpleNamespace())


class FakeBigVGAN(torch.nn.Module):
    removed_weight_norm = False

    @classmethod
    def from_pretrained(cls, model_name, use_cuda_kernel=False):
        model = cls()
        model.model_name = model_name
        model.use_cuda_kernel = use_cuda_kernel
        return model

    def remove_weight_norm(self):
        self.removed_weight_norm = True

    def decode(self, mel):
        return mel.mean(dim=1, keepdim=True).repeat_interleave(512, dim=-1)


class ExperimentalUpgradeTests(unittest.TestCase):
    def test_contentvec_layer_merge_reduces_three_layers_to_single_vec768_stream(self):
        from models import ContentVecLayerMerge

        merge = ContentVecLayerMerge(cv_dim=768, cv_layers=3)
        stacked = torch.randn(2, 768 * 3, 5)

        merged = merge(stacked)

        self.assertEqual((2, 768, 5), tuple(merged.shape))
        self.assertEqual((3,), tuple(merge.cv_layer_w.shape))

    def test_bigvgan_decoder_uses_mel_projection_and_keeps_decoder_call_contract(self):
        fake_module = types.SimpleNamespace(BigVGAN=FakeBigVGAN)
        sys.modules["bigvgan"] = fake_module
        try:
            from vdecoder.bigvgan import BigVGANDecoder

            decoder = BigVGANDecoder(
                latent_channels=192,
                n_mel_channels=100,
                model_name="nvidia/bigvgan_v2_44khz_128band_512x",
                trainable=False,
                use_cuda_kernel=True,
            )
            z = torch.randn(2, 192, 7)

            audio = decoder(z, g=torch.randn(2, 768, 7), f0=torch.randn(2, 7))

            self.assertEqual((2, 1, 3584), tuple(audio.shape))
            self.assertFalse(any(p.requires_grad for p in decoder.vocoder.parameters()))
            self.assertTrue(decoder.vocoder.removed_weight_norm)
            self.assertEqual("nvidia/bigvgan_v2_44khz_128band_512x", decoder.vocoder.model_name)
        finally:
            sys.modules.pop("bigvgan", None)

    def test_discriminator_kinds_include_requested_experimental_discriminators(self):
        from models import discriminator_kinds
        from utils import HParams

        hps = HParams(use_cqt_disc=True, use_mrd_disc=True, use_mbd_disc=True)

        self.assertEqual(["mpd", "cqt", "mrd", "mbd"], discriminator_kinds(hps))

    def test_speaker_adversarial_head_reverses_content_gradient(self):
        from models import SpeakerAdversarialHead

        head = SpeakerAdversarialHead(hidden_channels=4, n_speakers=2, weight=1.0)
        content = torch.randn(2, 4, 3, requires_grad=True)
        mask = torch.ones(2, 1, 3)

        loss = head(content, mask).sum()
        loss.backward()
        reversed_grad = content.grad.detach().clone()

        content2 = content.detach().clone().requires_grad_(True)
        plain_loss = head.proj((content2 * mask).mean(dim=-1)).sum()
        plain_loss.backward()

        self.assertTrue(torch.allclose(reversed_grad, -content2.grad, atol=1e-6))

    def test_dpg_gui_exposes_experimental_training_options(self):
        from local_gui_dpg import config

        field_paths = {field[0] for field in config.CONFIG_FIELDS}

        self.assertIn("vec768l12mix", config.SPEECH_ENCODERS)
        self.assertEqual(768, config.ENCODER_DIM["vec768l12mix"])
        self.assertIn("model.vocoder_name", field_paths)
        self.assertIn("model.bigvgan_model", field_paths)
        self.assertIn("model.use_speaker_adversarial", field_paths)
        self.assertIn("train.c_speaker_adv", field_paths)


if __name__ == "__main__":
    unittest.main()
