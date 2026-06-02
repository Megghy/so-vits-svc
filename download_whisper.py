"""下载 Whisper 权重到 pretrain/（whisper+contentvec 组合编码器所需）。

OpenAI 官方 checkpoint 自带 {"dims", "model_state_dict"}，与 vencoder/WhisperPPGLarge.py 直接兼容。
WhisperPPGLarge 会按 dims.n_mels 自适配(v2=80, v3=128)，并优先加载 large-v3.pt。
用法:
    python download_whisper.py                  # 默认下 large-v3 到 pretrain/large-v3.pt
    python download_whisper.py --model large-v2 # 改下 v2
    python download_whisper.py --url <镜像URL> --out pretrain/large-v3.pt
"""

import argparse
import hashlib
import os
import urllib.request

MODELS = {
    "large-v3": (
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb",
    ),
    "large-v2": (
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
        "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
    ),
}


_last_mark = [-1]


def _progress(block_num, block_size, total):
    done = block_num * block_size
    mark = int(done / 50e6)  # 每 ~50MB 打印一次,避免日志刷屏
    if mark == _last_mark[0]:
        return
    _last_mark[0] = mark
    if total > 0:
        pct = min(100.0, done * 100.0 / total)
        print(f"下载中 {done/1e6:.0f}/{total/1e6:.0f} MB ({pct:.1f}%)", flush=True)
    else:
        print(f"下载中 {done/1e6:.0f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="large-v3")
    ap.add_argument("--url", default=None, help="镜像 URL，覆盖 --model 的默认源")
    ap.add_argument("--out", default=None, help="输出路径，默认 pretrain/<model>.pt")
    ap.add_argument("--no_verify", action="store_true", help="跳过 SHA256 校验")
    args = ap.parse_args()

    default_url, sha = MODELS[args.model]
    url = args.url or default_url
    out = args.out or os.path.join("pretrain", f"{args.model}.pt")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        print(f"已存在 {out}，跳过下载。")
        return

    tmp = out + ".part"
    print(f"从 {url}\n下载到 {out}")
    urllib.request.urlretrieve(url, tmp, _progress)
    print()

    if not args.no_verify and args.url is None:
        print("校验 SHA256…")
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            os.remove(tmp)
            raise SystemExit(f"SHA256 不匹配，下载损坏:\n  期望 {sha}\n  实际 {h.hexdigest()}")
        print("校验通过。")

    os.replace(tmp, out)
    print(f"完成 {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

