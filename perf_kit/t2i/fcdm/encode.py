"""Images + captions -> latent shards for perf_kit.t2i.fcdm.train (needs `pip install diffusers`).

    python -m perf_kit.t2i.fcdm.encode --images DIR --out shards/ --size 256 [--shard_rows 4096]

DIR/captions.jsonl holds one {"image": "relative/path.png", "caption": "..."} per line. Each image is resized so its
short side is --size and centre-cropped to a square, encoded with the SD-VAE Agate uses (stabilityai/sd-vae-ft-mse),
and its latent MEAN is stored unscaled (the trainer multiplies by 0.18215).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def square(im, size: int) -> torch.Tensor:
    from PIL import Image
    im = im.convert("RGB")
    s = size / min(im.size)
    im = im.resize((max(size, round(im.width * s)), max(size, round(im.height * s))), Image.BICUBIC)
    left, top = (im.width - size) // 2, (im.height - size) // 2
    im = im.crop((left, top, left + size, top + size))
    x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(size, size, 3).permute(2, 0, 1)
    return x.float() / 127.5 - 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="perf_kit.t2i.fcdm.encode")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--shard_rows", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    from diffusers import AutoencoderKL
    from PIL import Image
    from safetensors.torch import save_file
    root, out = Path(a.images), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(ln) for ln in (root / "captions.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    vae = AutoencoderKL.from_pretrained(a.vae).to(a.device).eval()
    for s0 in range(0, len(rows), a.shard_rows):
        part = rows[s0:s0 + a.shard_rows]
        lat = []
        for b0 in range(0, len(part), a.batch):
            px = []
            for r in part[b0:b0 + a.batch]:
                with Image.open(root / r["image"]) as im:
                    px.append(square(im, a.size))
            with torch.no_grad():
                lat.append(vae.encode(torch.stack(px).to(a.device)).latent_dist.mean.float().cpu())
        name = f"shard-{s0 // a.shard_rows:05d}"
        save_file({"latents": torch.cat(lat).to(torch.float16).contiguous()}, str(out / f"{name}.safetensors"))
        (out / f"{name}.jsonl").write_text("".join(json.dumps({"caption": r["caption"]}, ensure_ascii=False) + "\n"
                                                   for r in part), encoding="utf-8")
        print(f"{name}: {len(part)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
