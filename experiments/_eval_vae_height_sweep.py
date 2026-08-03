"""입력 height 를 올리면 VAE 복원이 정말 좋아지나 — 높이에 공정한 두 기준으로 측정.

  (A) self  : 복원 vs 그 높이의 입력             → 해당 스케일에서의 충실도
  (B) native: 복원을 원본 해상도로 되돌려 비교      → 진짜 디테일 보존량 (높이 간 비교에 공정)

결론(카드 「Going above height 64」 절 근거): 글자당 픽셀 수가 늘면 계속 좋아지므로
원본이 고해상도면 h>64 가 유리하고, 원본 native 를 넘겨 확대하면 서서히 악화한다.
단 Eruku 에 물릴 latent 는 8행(=h64) 이어야 한다.

사용: CUDA_VISIBLE_DEVICES=0 .venv/bin/python _eval_vae_height_sweep.py [--vae <경로/repo>]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim

from _common import TRAIN_FONTS, HW_DIR
from infer_show import render_in_font

TEXTS = ["형형색색 꽃들이 활짝 피어난 봄날의 공원", "2024년 10월 15일 서울특별시 강남구",
         "인공지능 모델의 한국어 생성 성능 평가", "밝은 햇살 붉은 노을 넓은 들판"]


def gray01(img):
    return torch.from_numpy(np.array(img.convert("L"), np.float32) / 255.0)[None, None]


@torch.no_grad()
def measure(vae, img, h, dev):
    """(self MSE, self SSIM, native MSE, native SSIM, latent shape)"""
    nat = gray01(img)
    g = img.convert("L")
    w = max(1, round(h * g.width / g.height))
    x01 = torch.from_numpy(np.array(g.resize((w, h), Image.BICUBIC), np.float32) / 255.)[None, None]
    pad = -w % 8
    x = F.pad(x01 * 2 - 1, (0, pad), value=1.0).repeat(1, 3, 1, 1)
    z = vae.encode(x.to(dev)).latent_dist.mode()
    rec = ((vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu() + 1) / 2)
    if pad:
        rec = rec[..., : rec.shape[-1] - pad]
    W, H = min(rec.shape[-1], x01.shape[-1]), min(rec.shape[-2], x01.shape[-2])
    a, b = x01[0, 0, :H, :W].numpy(), rec[0, 0, :H, :W].numpy()
    up = F.interpolate(rec, size=(nat.shape[-2], nat.shape[-1]), mode="bicubic",
                       align_corners=False).clamp(0, 1)
    a2, b2 = nat[0, 0].numpy(), up[0, 0].numpy()
    return (float(np.mean((a - b) ** 2)), float(ssim(a, b, data_range=1.0)),
            float(np.mean((a2 - b2) ** 2)), float(ssim(a2, b2, data_range=1.0)), tuple(z.shape[-2:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="HERIUN/emuru_vae_korean")
    ap.add_argument("--heights", type=int, nargs="+", default=[48, 64, 80, 96, 128, 160, 192])
    ap.add_argument("--render-size", type=int, default=200, help="폰트 렌더 크기(native 해상도 확보용)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae).eval().to(dev)

    fonts = sorted(TRAIN_FONTS.glob("*.ttf"))[:3]
    groups = [(f"폰트 렌더 {len(TEXTS)*len(fonts)}장",
               [Image.fromarray(render_in_font(f, t, size=args.render_size)) for t in TEXTS for f in fonts])]
    scans = [Image.open(p) for p in sorted(HW_DIR.glob("line_*.png"))[:10]]
    if scans:
        groups.append((f"실제 스캔 {len(scans)}장", scans))

    for label, group in groups:
        print(f"\n=== {label} (native 예: {group[0].size}) ===")
        print(f"{'h':>5} {'latent':>10} | self MSE  self SSIM | native MSE  native SSIM")
        for h in args.heights:
            r = np.array([measure(vae, im, h, dev)[:4] for im in group], dtype=float).mean(0)
            lat = measure(vae, group[0], h, dev)[4]
            print(f"{h:5d} {str(lat):>10} | {r[0]:8.4f} {r[1]:9.3f} | {r[2]:10.4f} {r[3]:11.3f}"
                  + ("  ← 학습 해상도 / Eruku 요구" if h == 64 else ""))


if __name__ == "__main__":
    main()
