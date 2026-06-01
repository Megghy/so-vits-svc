"""Fit the Eta-WavLM speaker-removal projection W.

Eta-WavLM (ACL 2025, arXiv:2505.19273) linearly decomposes WavLM-Large features X
into a speaker-dependent component predictable from an utterance-level x-vector s and a
speaker-independent residual. With bias b = [1, s] (per utterance, constant over time):

    min_W  sum_{u,t} || X_{u,t} - b_u @ W ||^2
    =>  G @ W = H,  G = sum_u T_u * outer(b_u, b_u),  H = sum_u outer(b_u, sum_t X_{u,t})
    =>  W = G^{-1} H                                   (shape [spk_dim+1, hidden_dim])

IMPORTANT: run this on a MULTI-SPEAKER corpus (many speakers). The projection encodes
"where speaker identity lives" in WavLM space; it is dataset-agnostic and reused across
projects. Fitting on a single-speaker singing set is ill-posed (only removes the mean).

Usage:
    python eta_wavlm_fit.py --in_dir <multi_speaker_wav_dir> --out pretrain/eta_wavlm_proj.pt
"""

import argparse
import os
from glob import glob

import librosa
import torch
from tqdm import tqdm

from vencoder.eta_speaker import SpeakerEmbedder
from vencoder.WavLMLarge import WavLMLarge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="multi-speaker wav dir (searched recursively)")
    ap.add_argument("--wavlm_path", default="pretrain/WavLM-Large.pt")
    ap.add_argument("--spk_path", default="microsoft/wavlm-base-plus-sv",
                    help="WavLMForXVector model id or local dir")
    ap.add_argument("--out", default="pretrain/eta_wavlm_proj.pt")
    ap.add_argument("--output_layer", type=int, default=6)
    ap.add_argument("--ridge", type=float, default=1e-4,
                    help="Tikhonov ridge (scaled by mean diagonal) for numerical stability")
    ap.add_argument("--max_files", type=int, default=0, help="0 = use all")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    content = WavLMLarge(vec_path=args.wavlm_path, device=device, output_layer=args.output_layer)
    spk = SpeakerEmbedder(model_path=args.spk_path, device=device)
    hidden_dim = content.hidden_dim
    spk_dim = spk.dim

    files = glob(os.path.join(args.in_dir, "**", "*.wav"), recursive=True)
    files.sort()
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"No .wav found under {args.in_dir}")
    print(f"Fitting Eta-WavLM proj on {len(files)} files | hidden={hidden_dim} spk={spk_dim}")

    G = torch.zeros(spk_dim + 1, spk_dim + 1, dtype=torch.float64)
    H = torch.zeros(spk_dim + 1, hidden_dim, dtype=torch.float64)
    used = 0
    for f in tqdm(files):
        wav, _ = librosa.load(f, sr=16000)
        if wav.size < 1600:  # < 0.1s
            continue
        wt = torch.from_numpy(wav).to(device)
        with torch.no_grad():
            X = content.encoder(wt)  # [1, hidden_dim, T]
        X = X.squeeze(0).transpose(0, 1).double().cpu()  # [T, hidden_dim]
        s = spk.embed(wt).double().cpu()  # [spk_dim]
        b = torch.cat([torch.ones(1, dtype=torch.float64), s])  # [spk_dim+1]
        G += X.shape[0] * torch.outer(b, b)
        H += torch.outer(b, X.sum(0))
        used += 1

    if used == 0:
        raise SystemExit("No usable audio (all too short).")
    G += args.ridge * G.diagonal().mean() * torch.eye(spk_dim + 1, dtype=torch.float64)
    W = torch.linalg.solve(G, H)  # [spk_dim+1, hidden_dim]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "W": W.float(),
        "spk_path": args.spk_path,
        "output_layer": args.output_layer,
        "hidden_dim": hidden_dim,
        "spk_dim": spk_dim,
        "n_files": used,
    }, args.out)
    print(f"Saved {args.out} | W{tuple(W.shape)} | files used {used}")


if __name__ == "__main__":
    main()
