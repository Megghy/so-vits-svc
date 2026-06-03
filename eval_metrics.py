import torch
from torch.nn import functional as F


DEFAULT_SPEAKER_SIMILARITY_MODEL = "microsoft/wavlm-base-plus-sv"
SPEAKER_SIMILARITY_SAMPLE_RATE = 16000


def prepare_speaker_waveform(audio, sample_rate, resample_fn=None):
    wav = audio.detach().to(dtype=torch.float32).flatten().cpu()
    if sample_rate == SPEAKER_SIMILARITY_SAMPLE_RATE:
        return wav
    if resample_fn is None:
        from torchaudio.functional import resample as resample_fn
    return resample_fn(wav, sample_rate, SPEAKER_SIMILARITY_SAMPLE_RATE)


def speaker_similarity(generated, reference, sample_rate, embedder, resample_fn=None):
    with torch.no_grad():
        generated_wav = prepare_speaker_waveform(generated, sample_rate, resample_fn)
        reference_wav = prepare_speaker_waveform(reference, sample_rate, resample_fn)
        generated_emb = embedder.embed(generated_wav).to(dtype=torch.float32).flatten()
        reference_emb = embedder.embed(reference_wav).to(dtype=torch.float32).flatten()
        score = F.cosine_similarity(generated_emb, reference_emb, dim=0).clamp(-1.0, 1.0)
    return float(score.item())


class SpeakerSimilarityMetric:
    def __init__(self, model_path=DEFAULT_SPEAKER_SIMILARITY_MODEL, device="cpu"):
        from vencoder.eta_speaker import SpeakerEmbedder

        self.embedder = SpeakerEmbedder(model_path=model_path, device=device)

    def score(self, generated, reference, sample_rate):
        return speaker_similarity(generated, reference, sample_rate, self.embedder)


def collect_speaker_similarity_scalars(audio_pairs, sample_rate, metric):
    scalars = {}
    scores = []
    for index, (generated, reference) in enumerate(audio_pairs):
        score = metric.score(generated, reference, sample_rate)
        scores.append(score)
        scalars[f"eval/speaker_similarity/audio_{index}"] = score
    if not scores:
        return scalars
    mean_score = sum(scores) / len(scores)
    scalars["eval/speaker_similarity/mean"] = mean_score
    scalars["eval/speaker_distance/mean"] = 1.0 - mean_score
    return scalars
