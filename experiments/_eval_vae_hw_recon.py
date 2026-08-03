"""실제 손글씨(custom_datasets/hw_line_sample) 복원으로 VAE 일반화 비교.

원본 emuru VAE(10만 폰트 학습) vs 한글적응 decoder/full_mse VAE(우리 99폰트). full_mse 가 우리 폰트에
과적합해 **분포 밖(실제 스캔 손글씨) 복원이 오히려 나빠졌는지**를 roundtrip MSE/SSIM 로 측정 + montage.
폰트 렌더 복원(_eval_vae_recon.py)과 상보적 — 여긴 진짜 스캔 이미지가 입력.

예:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python _eval_vae_hw_recon.py \
    --models "원본=blowing-up-groundhogs/emuru_vae" \
             "decoder=finetune_runs/vae_korean_dec/vae_s15000" \
             "full_mse=finetune_runs/vae_korean_full_mse/vae_s28000"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim

HERE = Path(__file__).resolve().parent
HW_DIR = HERE / "custom_datasets" / "hw_line_sample"
LABEL_FONT = HERE / "assets/fonts_label/NanumGothic-Regular.ttf"
H = 64


def load_hw(p, h=H):
    """실제 손글씨 PNG → clean-ish [1,1,64,W] [-1,1] (bg=+1, ink=-1). 폭 8배수 우측 흰패딩."""
    im = Image.open(p).convert("L")
    a = np.array(im).astype(np.float32) / 255.0
    t = torch.from_numpy(a)[None, None]
    w64 = max(1, int(round(h * a.shape[1] / a.shape[0])))
    t = F.interpolate(t, size=(h, w64), mode="bilinear", align_corners=False).clamp(0, 1)
    x = t * 2 - 1
    w8 = (w64 + 7) // 8 * 8
    if w8 != w64:
        x = F.pad(x, (0, w8 - w64), value=1.0)
    return x


@torch.no_grad()
def roundtrip(vae, x, dev):
    z = vae.encode(x.repeat(1, 3, 1, 1).to(dev)).latent_dist.mode()
    return vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu()


def metrics(orig, rec):
    a = ((orig[0, 0] + 1) / 2).numpy(); b = ((rec[0, 0] + 1) / 2).numpy()
    return float(np.mean((a - b) ** 2)), float(ssim(a, b, data_range=1.0))


def to_u8(t):
    return ((t[0, 0] + 1) / 2 * 255).clamp(0, 255).byte().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help='"라벨=경로" 목록')
    ap.add_argument("--out", default=str(HERE / "finetune_runs/eruku_short_fullmse/_eval/vaegen_hw.png"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    specs = [m.split("=", 1) for m in args.models]
    vaes = [(lbl, AutoencoderKL.from_pretrained(path).to(dev).eval()) for lbl, path in specs]
    lab = ImageFont.truetype(str(LABEL_FONT), 15) if LABEL_FONT.exists() else ImageFont.load_default()
    hdrf = ImageFont.truetype(str(LABEL_FONT), 17) if LABEL_FONT.exists() else ImageFont.load_default()

    labels = {}
    for ln in (HW_DIR / "label.txt").read_text(encoding="utf-8").splitlines():
        if "\t" in ln:
            fn, tx = ln.split("\t", 1); labels[fn.strip()] = tx.strip()
    files = sorted(HW_DIR.glob("line_*.png"))
    idx = np.linspace(0, len(files) - 1, args.n).round().astype(int).tolist()
    files = [files[i] for i in sorted(set(idx))]

    LW = 120
    agg = {lbl: {"m": [], "s": []} for lbl, _ in vaes}
    blocks = []
    for p in files:
        x = load_hw(p); w = x.shape[-1]
        def strip(t_u8, name, extra=""):
            row = np.full((H, LW + w), 255, np.uint8)
            row[:, LW:LW + t_u8.shape[1]] = t_u8
            im = Image.fromarray(row); d = ImageDraw.Draw(im)
            d.text((6, 4), name, font=lab, fill=0)
            if extra: d.text((6, H - 22), extra, font=lab, fill=0)
            return np.array(im)
        head = Image.new("L", (LW + w, 22), 235)
        ImageDraw.Draw(head).text((6, 2), f"{p.name}: {labels.get(p.name,'')}", font=hdrf, fill=0)
        parts = [np.array(head), strip(to_u8(x), "실제 손글씨")]
        for lbl, vae in vaes:
            r = roundtrip(vae, x, dev); m, s = metrics(x, r)
            agg[lbl]["m"].append(m); agg[lbl]["s"].append(s)
            parts += [np.full((2, LW + w), 210, np.uint8), strip(to_u8(r), lbl, f"MSE {m:.4f} SSIM {s:.3f}")]
        parts.append(np.full((8, LW + w), 90, np.uint8))
        blocks.append(np.vstack(parts))

    Wm = max(b.shape[1] for b in blocks)
    blocks = [np.hstack([b, np.full((b.shape[0], Wm - b.shape[1]), 255, np.uint8)]) if b.shape[1] < Wm else b
              for b in blocks]
    base = float(np.mean(agg[vaes[0][0]]["m"]))
    seg = []
    for lbl, _ in vaes:
        mm = float(np.mean(agg[lbl]["m"])); ss = float(np.mean(agg[lbl]["s"]))
        if lbl == vaes[0][0]:
            seg.append(f"{lbl} {mm:.4f}/{ss:.3f}")
        else:
            d = (mm / base - 1) * 100
            seg.append(f"{lbl} {mm:.4f}/{ss:.3f}({'+' if d > 0 else ''}{d:.0f}%)")
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
