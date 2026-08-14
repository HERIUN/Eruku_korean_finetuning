"""실제 손글씨(custom_datasets/hw_line_sample) 복원으로 VAE 일반화 비교.

원본 emuru VAE(10만 폰트 학습) vs 한글적응 decoder/full_mse VAE(우리 99폰트). full_mse 가 우리 폰트에
과적합해 **분포 밖(실제 스캔 손글씨) 복원이 오히려 나빠졌는지**를 roundtrip MSE/SSIM 로 측정 + montage.
폰트 렌더 복원(vae_recon.py)과 상보적 — 여긴 진짜 스캔 이미지가 입력.

예:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python vae_hw_recon.py \
    --models "원본=blowing-up-groundhogs/emuru_vae" \
             "decoder=finetune_runs/vae_korean_dec/vae_s15000" \
             "full_mse=finetune_runs/vae_korean_full_mse/vae_s28000"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL

from common import (REPO, HW_DIR, load_line_x11, read_hw_labels,
                     roundtrip, metrics, to_u8, label_fonts, strip_row, pad_blocks, vae_summary)

H = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help='"라벨=경로" 목록')
    ap.add_argument("--out", default=str(REPO / "finetune_runs/eruku_short_fullmse/_eval/vaegen_hw.png"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    specs = [m.split("=", 1) for m in args.models]
    vaes = [(lbl, AutoencoderKL.from_pretrained(path).to(dev).eval()) for lbl, path in specs]
    lab, hdrf = label_fonts()

    lines = read_hw_labels(HW_DIR)
    idx = np.linspace(0, len(lines) - 1, args.n).round().astype(int).tolist()
    sel = [lines[i] for i in sorted(set(idx))]

    LW = 120
    agg = {lbl: {"m": [], "s": []} for lbl, _ in vaes}
    blocks = []
    for p, txt in sel:
        x = load_line_x11(p); w = x.shape[-1]
        head = Image.new("L", (LW + w, 22), 235)
        ImageDraw.Draw(head).text((6, 2), f"{p.name}: {txt}", font=hdrf, fill=0)
        parts = [np.array(head), strip_row(to_u8(x), "실제 손글씨", w, LW, H, lab)]
        for lbl, vae in vaes:
            r = roundtrip(vae, x, dev); m, s = metrics(x, r)
            agg[lbl]["m"].append(m); agg[lbl]["s"].append(s)
            parts += [np.full((2, LW + w), 210, np.uint8),
                      strip_row(to_u8(r), lbl, w, LW, H, lab, f"MSE {m:.4f} SSIM {s:.3f}")]
        parts.append(np.full((8, LW + w), 90, np.uint8))
        blocks.append(np.vstack(parts))

    Wm, blocks = pad_blocks(blocks)
    seg = vae_summary(agg, [lbl for lbl, _ in vaes])
    summ = Image.new("L", (Wm, 30), 255)
    ImageDraw.Draw(summ).text((6, 6), "실제 손글씨 VAE 복원 일반화 (roundtrip, MSE↓/SSIM↑)  평균  " + "  |  ".join(seg),
                              font=hdrf, fill=0)
    out = np.vstack([np.array(summ), np.full((3, Wm), 0, np.uint8)] + blocks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}")
    print("  |  ".join(seg))


if __name__ == "__main__":
    main()
