import torch

from vencoder.ContentVec768L12 import ContentVec768L12
from vencoder.encoder import SpeechEncoder
from vencoder.WhisperPPGLarge import WhisperPPGLarge

# 双编码器:Whisper-PPG-Large(强解耦语言内容) + ContentVec 多层(保留韵律/表现力)。
# ContentVec 取 L6/L9/L12 三层在 channel 维拼接,交给模型里的 ContentMerge 做可学习加权。
# 输出顺序固定为 [Whisper 1280 | CV_L6 768 | CV_L9 768 | CV_L12 768] = 3584 通道,
# 帧率两分支均为 50fps(16k/320),长度差超过 2 帧直接报错。ContentMerge 见 models.py。
CV_LAYERS = (6, 9, 12)
WHISPER_DIM = 1280
CV_DIM = 768


class WhisperContentVec(SpeechEncoder):
    def __init__(self, whisper_path="pretrain/large-v3.pt", device=None):
        super().__init__()
        self.whisper = WhisperPPGLarge(vec_path=whisper_path, device=device)
        self.cv = ContentVec768L12(device=device)
        self.dev = self.cv.dev
        self.cv_layers = CV_LAYERS
        self.hidden_dim = WHISPER_DIM + CV_DIM * len(CV_LAYERS)

    def _cv_layers(self, wav16k):
        feats = wav16k
        if feats.dim() == 2:
            feats = feats.mean(-1)
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        source = feats.to(self.dev)
        padding_mask = padding_mask.to(self.dev)
        with torch.no_grad():
            features = self.cv.model.forward_features(source)
            features = features.transpose(1, 2)
            features = self.cv.model.layer_norm(features)
            if padding_mask is not None:
                padding_mask = self.cv.model.forward_padding_mask(features, padding_mask)
            if self.cv.model.post_extract_proj is not None:
                features = self.cv.model.post_extract_proj(features)
            x = self.cv.model.dropout_input(features)
            _, layer_results = self.cv.model.encoder(x, padding_mask=padding_mask, layer=None)
        if len(layer_results) < max(self.cv_layers):
            raise RuntimeError(f"ContentVec returned {len(layer_results)} layers, need {max(self.cv_layers)}")
        return [
            layer_results[layer - 1][0].transpose(0, 1).transpose(1, 2)
            for layer in self.cv_layers
        ]

    @staticmethod
    def _align_features(features):
        lengths = [f.shape[-1] for f in features]
        max_len = max(lengths)
        min_len = min(lengths)
        if max_len - min_len > 2:
            raise RuntimeError(f"whisper+contentvec frame length mismatch: {lengths}")
        return [f[..., :min_len] for f in features]

    def encoder(self, wav):
        w = self.whisper.encoder(wav)  # [1, 1280, Tw]
        cvs = self._cv_layers(wav)  # 3 × [1, 768, Tc]
        parts = self._align_features([w.to(self.dev)] + cvs)
        return torch.cat(parts, dim=1)  # [1, 3584, t]
