import torch

from vencoder.ContentVec768L12 import ContentVec768L12


CV_LAYERS = (6, 9, 12)
CV_DIM = 768


class ContentVec768LayerMix(ContentVec768L12):
    def __init__(self, vec_path="pretrain/checkpoint_best_legacy_500.pt", device=None):
        super().__init__(vec_path=vec_path, device=device)
        self.cv_layers = CV_LAYERS
        self.hidden_dim = CV_DIM * len(CV_LAYERS)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        source = feats.to(wav.device)
        padding_mask = padding_mask.to(wav.device)

        with torch.no_grad():
            features = self.model.forward_features(source)
            features = features.transpose(1, 2)
            features = self.model.layer_norm(features)
            padding_mask = self.model.forward_padding_mask(features, padding_mask)
            if self.model.post_extract_proj is not None:
                features = self.model.post_extract_proj(features)
            x = self.model.dropout_input(features)
            _, layer_results = self.model.encoder(x, padding_mask=padding_mask, layer=None)

        if len(layer_results) < max(self.cv_layers):
            raise RuntimeError(f"ContentVec returned {len(layer_results)} layers, need {max(self.cv_layers)}")
        return torch.cat(
            [
                layer_results[layer - 1][0].transpose(0, 1).transpose(1, 2)
                for layer in self.cv_layers
            ],
            dim=1,
        )
