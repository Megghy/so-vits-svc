def get_hparam(h, key, default):
    if isinstance(h, dict):
        return h.get(key, default)
    return getattr(h, key, default)


def get_source_scale(h):
    return float(get_hparam(h, "nsf_source_scale", 1.0))


def build_source_stats(har_source, uv, source_scale):
    uv_mask = uv.transpose(1, 2).float()
    voiced = uv_mask.sum().clamp_min(1.0)
    unvoiced = (1.0 - uv_mask).sum().clamp_min(1.0)
    src_sq = har_source.float().pow(2)
    return {
        "source_scale": source_scale,
        "source_rms": src_sq.mean().sqrt().item(),
        "source_rms_voiced": (src_sq * uv_mask).sum().div(voiced).sqrt().item(),
        "source_rms_unvoiced": (src_sq * (1.0 - uv_mask)).sum().div(unvoiced).sqrt().item(),
        "voiced_ratio": uv_mask.mean().item(),
    }


def add_injection_stats(stats, stage, injected_source, carrier):
    injected_power = injected_source.float().pow(2).mean()
    carrier_power = carrier.detach().float().pow(2).mean().clamp_min(1e-8)
    stats[f"source_injection_rms_{stage}"] = injected_power.sqrt().item()
    stats[f"source_injection_ratio_{stage}"] = (injected_power / carrier_power).sqrt().item()
