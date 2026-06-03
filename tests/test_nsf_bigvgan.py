import unittest
from unittest.mock import patch

import torch
import torch.nn as nn


class FakeAMPBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class FakeBigVGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_kernels = 1
        self.num_upsamples = 2
        self.conv_pre = nn.Conv1d(128, 4, 3, padding=1)
        self.ups = nn.ModuleList([
            nn.ModuleList([nn.ConvTranspose1d(4, 2, 4, stride=2, padding=1)]),
            nn.ModuleList([nn.ConvTranspose1d(2, 1, 4, stride=2, padding=1)]),
        ])
        self.resblocks = nn.ModuleList([FakeAMPBlock(2), FakeAMPBlock(1)])
        self.activation_post = nn.Identity()
        self.conv_post = nn.Conv1d(1, 1, 3, padding=1)
        self.use_tanh_at_final = True

    def remove_weight_norm(self):
        pass


def fake_from_pretrained(**kwargs):
    return FakeBigVGAN()


class NSFBigVGANTests(unittest.TestCase):
    def test_generator_forward_uses_nested_bigvgan_upsamplers(self):
        import bigvgan
        from vdecoder.nsf_bigvgan import NSFBigVGANGenerator

        with patch.object(bigvgan.BigVGAN, "_from_pretrained", staticmethod(fake_from_pretrained)):
            generator = NSFBigVGANGenerator(trainable=False)

        self.assertEqual(generator._infer_upsample_rates(), [2, 2])
        self.assertEqual(generator.upp, 4)

        mel = torch.randn(2, 128, 5)
        f0 = torch.full((2, 5), 220.0)
        with torch.no_grad():
            audio = generator(mel, f0)

        self.assertEqual(tuple(audio.shape), (2, 1, 20))
        self.assertLessEqual(float(audio.abs().max()), 1.0)

    def test_f0_source_changes_output_with_same_mel(self):
        import bigvgan
        from vdecoder.nsf_bigvgan import NSFBigVGANGenerator

        with patch.object(bigvgan.BigVGAN, "_from_pretrained", staticmethod(fake_from_pretrained)):
            generator = NSFBigVGANGenerator(trainable=False)

        for conv in generator.noise_convs:
            nn.init.constant_(conv.weight, 0.1)
            nn.init.zeros_(conv.bias)

        mel = torch.randn(1, 128, 8)
        voiced_f0 = torch.full((1, 8), 220.0)
        unvoiced_f0 = torch.zeros(1, 8)

        # 门控初始为 0：源注入为零，f0 不应改变输出(严格等价官方 BigVGAN)
        with torch.no_grad():
            torch.manual_seed(1234)
            gated_voiced = generator(mel, voiced_f0)
            torch.manual_seed(1234)
            gated_unvoiced = generator(mel, unvoiced_f0)
        self.assertTrue(torch.allclose(gated_voiced, gated_unvoiced))

        # 打开门控后：source 路径生效，f0 应改变输出
        for gain in generator.source_gains:
            nn.init.constant_(gain, 1.0)

        with torch.no_grad():
            torch.manual_seed(1234)
            voiced_audio = generator(mel, voiced_f0)
            torch.manual_seed(1234)
            unvoiced_audio = generator(mel, unvoiced_f0)

        self.assertFalse(torch.allclose(voiced_audio, unvoiced_audio))

    def test_source_gain_initialized_to_zero(self):
        """source gate 初始必须全为 0：保证训练起点等价官方 BigVGAN。"""
        import bigvgan
        from vdecoder.nsf_bigvgan import NSFBigVGANGenerator

        with patch.object(bigvgan.BigVGAN, "_from_pretrained", staticmethod(fake_from_pretrained)):
            generator = NSFBigVGANGenerator(trainable=False)

        self.assertEqual(len(generator.source_gains), generator.num_upsamples)
        for gain in generator.source_gains:
            self.assertTrue(gain.requires_grad)
            self.assertEqual(float(gain), 0.0)

    def test_decoder_forward_and_phase_transition(self):
        import bigvgan
        from vdecoder.nsf_bigvgan import NSFBigVGANDecoder

        with patch.object(bigvgan.BigVGAN, "_from_pretrained", staticmethod(fake_from_pretrained)):
            decoder = NSFBigVGANDecoder(latent_channels=8, gin_channels=4, trainable=False)

        self.assertFalse(decoder.trainable)
        self.assertFalse(next(decoder.vocoder.bigvgan_base.parameters()).requires_grad)
        self.assertTrue(next(decoder.vocoder.m_source.parameters()).requires_grad)

        z = torch.randn(1, 8, 6)
        g = torch.randn(1, 4, 1)
        f0 = torch.full((1, 6), 220.0)
        with torch.no_grad():
            audio, pred_mel = decoder(z, g, f0)

        self.assertEqual(tuple(pred_mel.shape), (1, 128, 6))
        self.assertEqual(tuple(audio.shape), (1, 1, 24))

        decoder.unfreeze_vocoder()
        self.assertTrue(decoder.trainable)
        self.assertTrue(next(decoder.vocoder.bigvgan_base.parameters()).requires_grad)

    def test_synthesizer_selects_nsf_bigvgan_vocoder(self):
        import bigvgan
        from models import SynthesizerTrn

        with patch.object(bigvgan.BigVGAN, "_from_pretrained", staticmethod(fake_from_pretrained)):
            net = SynthesizerTrn(
                spec_channels=513,
                segment_size=4,
                inter_channels=8,
                hidden_channels=8,
                filter_channels=16,
                n_heads=2,
                n_layers=1,
                kernel_size=3,
                p_dropout=0.0,
                resblock="1",
                resblock_kernel_sizes=[3],
                resblock_dilation_sizes=[[1, 3, 5]],
                upsample_rates=[2, 2],
                upsample_initial_channel=16,
                upsample_kernel_sizes=[4, 4],
                gin_channels=4,
                ssl_dim=8,
                n_speakers=2,
                vocoder_name="nsf-bigvgan-v2",
            )

        from vdecoder.nsf_bigvgan import NSFBigVGANDecoder

        self.assertIsInstance(net.dec, NSFBigVGANDecoder)


if __name__ == "__main__":
    unittest.main()
