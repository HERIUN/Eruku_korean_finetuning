"""실제 손글씨 라인(custom_datasets/hw_line_sample)을 style 로 주고 같은 텍스트를 생성하는
**미러링(echo)** 성능을 모델별로 비교하는 montage.

미러링 = 실제 손글씨 라인 이미지를 조건(style)으로, 그 라인의 전사 텍스트를 그대로 생성 →
모델이 (a)텍스트를 정확히 쓰고 (b)그 손글씨 스타일을 얼마나 재현하는지. GT = 실제 손글씨 자체.

행 = 라인샘플(짧은→긴), 열 = [실제 손글씨(=목표) | 모델1 echo | 모델2 echo | …].
모델은 파이프라인 단계별: 기존(원본VAE) → decoderVAE교체 → full_mse재적응前 → 현재최적.

예:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python mirror_compare.py \
    --out finetune_runs/eruku_short_fullmse/_eval/mirror_models.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image
import torch.nn.functional as F

from common import REPO, HW_DIR, load_line_x11, read_hw_labels, header_row, fit_row
from infer.show import load_model, load_style, gen_from_style, cell, label_img


def bbox_crop_gray(p, thr=180, bthr=215, dens_col=0.03, dens_row=0.006, solid=0.55, pad=4):
    """실제 손글씨 PNG → ink bbox 로 타이트 크롭한 grayscale [H,W] uint8.
    여백/마진이 인코더 불확실성(latent std↑)을 유발해 generation runaway → bbox 크롭으로 제거.
    2단계: ① 가장자리 solid 테두리(스캔 프레임, 밀도>solid 인 바깥 행/열) 벗겨냄
           ② 밀도 기반 bbox — sparse 노이즈 무시, 여백 제거.
    ★임계 비대칭: 행(세로)은 낮게(dens_row) — 글자 위아래 성긴 획(받침·삐침)까지 살려 안 잘리게.
      열(가로)은 dens_col — 좌우 여백 제거. 테두리 감지는 회색까지(bthr<215), 텍스트는 진하게(thr<180)."""
    a = np.array(Image.open(p).convert("L"))
    bink = a < bthr                                    # 테두리 감지(회색 포함)
    y0, y1, x0, x1 = 0, a.shape[0], 0, a.shape[1]
    while y0 < y1 and bink[y0].mean() > solid: y0 += 1
    while y1 > y0 and bink[y1 - 1].mean() > solid: y1 -= 1
    while x0 < x1 and bink[:, x0].mean() > solid: x0 += 1
    while x1 > x0 and bink[:, x1 - 1].mean() > solid: x1 -= 1
    a = a[y0:y1, x0:x1]
    ink = a < thr                                      # 텍스트 밀도
    cold, rowd = ink.mean(0), ink.mean(1)
    cols, rows = np.where(cold > dens_col)[0], np.where(rowd > dens_row)[0]
    if len(cols) == 0 or len(rows) == 0:
        return a
    yy0, yy1 = max(0, rows.min() - pad), min(a.shape[0], rows.max() + pad + 1)
    xx0, xx1 = max(0, cols.min() - pad), min(a.shape[1], cols.max() + pad + 1)
    return a[yy0:yy1, xx0:xx1]


def style_from_gray(a_u8):
    """grayscale [H,W] uint8 → style tensor [3,64,W] [-1,1] (load_style 와 동일 규약, 패딩 없음)."""
    return load_line_x11(a_u8, pad8=False)[0].repeat(3, 1, 1)

# (라벨, ckpt, vae_checkpoint) — 파이프라인 단계별. vae=None 이면 ckpt 내장 VAE 사용.
MODELS = [
    ("기존 (원본 VAE)",        "finetune_runs/korean_en2ko_fixed/checkpoint_step_011000.pth", None),
    ("decoder VAE 교체",       "finetune_runs/korean_en2ko_fixed/checkpoint_step_011000.pth",
                               "finetune_runs/vae_korean_dec/vae_s15000"),
    ("full_mse 재적응前 s15k",  "finetune_runs/eruku_readapt_fullvae/checkpoint_step_015000.pth", None),
    ("현재 최적 s30k",          "finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "finetune_runs/eruku_short_fullmse/_eval/mirror_models.png"))
    ap.add_argument("--cfg", type=float, default=1.25)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-w", type=int, default=440)
    ap.add_argument("--cell-h", type=int, default=76)
    ap.add_argument("--rows", nargs="*", type=int, default=None,
                    help="사용할 라인 인덱스(기본: 다양 길이 자동 선택)")
    ap.add_argument("--bbox-crop", dest="bbox_crop", action="store_true", default=True,
                    help="style 을 ink bbox 로 타이트 크롭 후 인코딩(여백 제거→latent std↓→runaway 방지). 기본 on")
    ap.add_argument("--no-bbox-crop", dest="bbox_crop", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    CW, CH = args.cell_w, args.cell_h

    lines = read_hw_labels(HW_DIR)
    if not lines:
        raise SystemExit(f"라인샘플 없음: {HW_DIR}")
    # 짧은→긴 다양 길이로 8개 선택(균등 인덱스)
    idx = args.rows if args.rows else sorted(set(np.linspace(0, len(lines) - 1, 8).round().astype(int).tolist()))
    sel = [lines[i] for i in idx]
    print(f"selected {len(sel)} lines: {[t[:12] for _, t in sel]}")

    models = []
    for lbl, ck, vae in MODELS:
        print(f"loading [{lbl}] {ck}" + (f"  +VAE {vae}" if vae else ""))
        m, step = load_model(ck, device, vae_checkpoint=vae)
        models.append((lbl, m, step))

    tag = "bbox크롭" if args.bbox_crop else "raw"
    col_labels = [f"실제 손글씨 (목표=style, {tag})"] + [lbl for lbl, _, _ in models]
    ncol = len(col_labels); full_w = ncol * (CW + 4)

    def style_of(style_path):
        """bbox 크롭 여부에 따라 style tensor 반환."""
        if args.bbox_crop:
            return style_from_gray(bbox_crop_gray(style_path))
        return load_style(style_path)

    def echo(model, style_path, text):
        return gen_from_style(model, style_of(style_path), text, text,
                              args.cfg, args.max_new_tokens, device,
                              max_img_len=4096, seed=args.seed)        # 모델 간 동일 조건

    grid = []
    hdr = header_row(col_labels, CW, CH)
    grid.append(hdr); grid.append(np.full((3, hdr.shape[1]), 80, np.uint8))
    for style_path, text in sel:
        hw = bbox_crop_gray(style_path) if args.bbox_crop else np.array(Image.open(style_path).convert("L"))
        # 셀에 flush 로 붙어 잘려 보이지 않게 좌우 흰 여백(폭의 6%) 추가
        mgn = max(6, int(hw.shape[1] * 0.06))
        hw = np.pad(hw, ((2, 2), (mgn, mgn)), constant_values=255)
        cells = [cell(hw, f"목표: {text[:26]}", CW, CH, highlight=True)]
        for lbl, m, _ in models:
            g = echo(m, style_path, text)
            gm = max(6, int(g.shape[1] * 0.06))
            g = np.pad(g, ((2, 2), (gm, gm)), constant_values=255)
            cells.append(cell(g, lbl, CW, CH))
        row = fit_row(cells, full_w)
        grid.append(row); grid.append(np.full((6, row.shape[1]), 80, np.uint8))
        print(f"  {style_path.name}: {text[:30]!r}")
    body = np.vstack(grid); Wd = body.shape[1]
    steps = " / ".join(f"{lbl}=s{st}" for lbl, _, st in models)
    title = label_img(f'실제 손글씨 미러링(echo) 모델별 비교  (cfg={args.cfg})  '
                      f'← 실제 손글씨=조건 | 각 모델이 같은 텍스트를 그 스타일로 재현 →   [{steps}]',
                      Wd, 44, 17, bold=True, center=False, bg=255)
    out = np.vstack([title, np.full((2, Wd), 0, np.uint8), body])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
