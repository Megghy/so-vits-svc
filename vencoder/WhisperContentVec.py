import torch

from vencoder.ContentVec768L12 import ContentVec768L12
from vencoder.encoder import SpeechEncoder
from vencoder.WhisperPPGLarge import WhisperPPGLarge

# 双编码器:Whisper-PPG-Large(强解耦语言内容) + ContentVec 多层(保留韵律/表现力)。
# ContentVec 取 L6/L9/L12 三层在 channel 维拼接,交给模型里的 ContentMerge 做可学习加权。
# 输出顺序固定为 [Whisper 1280 | CV_L6 768 | CV_L9 768 | CV_L12 768] = 3584 通道,
# 帧率两分支均为 50fps(16k/320),取 min 对齐。ContentMerge 见 models.py。
CV_LAYERS = (6, 9, 12)
WHISPER_DIM = 1280
CV_DIM = 768


class WhisperContentVec(SpeechEncoder):
    def __init__(self, device=None):
        super().__init__()
        self.whisper = WhisperPPGLarge(device=device)
        self.cv = ContentVec768L12(device=device)
        self.dev = self.cv.dev
        self.cv_layers = CV_LAYERS
        self.hidden_dim = WHISPER_DIM + CV_DIM * len(CV_LAYERS)

    def _cv_layer(self, wav16k, layer):
        feats = wav16k
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(self.dev),
            "padding_mask": padding_mask.to(self.dev),
            "output_layer": layer,
        }
        with torch.no_grad():
            logits = self.cv.model.extract_features(**inputs)
        return logits[0].transpose(1, 2)  # [1, 768, T]

    def encoder(self, wav):
        w = self.whisper.encoder(wav)  # [1, 1280, Tw]
        cvs = [self._cv_layer(wav, layer) for layer in self.cv_layers]  # 3 × [1, 768, Tc]
        t = min([w.shape[-1]] + [c.shape[-1] for c in cvs])
        parts = [w[..., :t].to(self.dev)] + [c[..., :t] for c in cvs]
        return torch.cat(parts, dim=1)  # [1, 3584, t]
