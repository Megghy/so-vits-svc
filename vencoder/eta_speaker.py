import numpy as np
import torch


class SpeakerEmbedder:
    """Utterance-level speaker embedding (x-vector) used by Eta-WavLM.

    Wraps transformers' WavLMForXVector (microsoft/wavlm-base-plus-sv).
    Input: 16kHz mono waveform. Output: L2-normalized [spk_dim] vector.
    transformers is already a project dependency, so no extra install needed.
    A local directory can be passed as model_path for offline use.
    """

    def __init__(self, model_path="microsoft/wavlm-base-plus-sv", device=None):
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector
        except Exception as e:
            raise ImportError(
                "Eta-WavLM speaker embedder needs transformers' WavLMForXVector. "
                "Upgrade transformers (pip install -U transformers)."
            ) from e
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = WavLMForXVector.from_pretrained(model_path).to(self.device).eval()
        self.dim = self.model.config.xvector_output_dim

    @torch.no_grad()
    def embed(self, wav16k):
        """wav16k: 1D tensor or np.ndarray at 16kHz. Returns [spk_dim] tensor on device."""
        if torch.is_tensor(wav16k):
            wav16k = wav16k.detach().to(torch.float32).cpu().numpy()
        wav16k = np.asarray(wav16k, dtype=np.float32)
        inputs = self.feature_extractor(wav16k, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = self.model(**inputs).embeddings  # [1, spk_dim]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0)
