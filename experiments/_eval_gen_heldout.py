"""Held-out 폰트(assets/custom_hw_fonts, 학습 미사용 AI 손글씨 4종)로 Eruku 생성 montage.

각 폰트를 style 조건으로, 단문/복잡획/장문 타겟을 생성 → unseen 손글씨-스타일 폰트 일반화 확인.
폰트-렌더(고대비)라 실제 스캔과 달리 runaway 없음. 열 = [style ref | 정답 | 기존 | 현재최적 s30k].

예:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python _eval_gen_heldout.py \
    --out finetune_runs/eruku_short_fullmse/_eval/gen_heldout_fonts.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image

from _common import REPO, header_row, fit_row
from infer_show import load_model, render_in_font, style_tensor, gen_from_style, cell, label_img
from eval_echo_metrics import font_can_render

FONTS_DIR = REPO / "assets" / "custom_hw_fonts"
STYLE_PHRASE = "한국어 손글씨 스타일 미리보기"      # 각 폰트의 style 조건(타겟과 다른 문장)
TARGETS = ["밝은 햇살", "붉은 노을 짧은 봄밤",
           "인공지능 모델 성능 평가",
           "형형색색 꽃들이 활짝 피어난 봄날의 공원"]

MODELS = [
    ("기존 (원본 VAE, s11k)", "finetune_runs/korean_en2ko_fixed/checkpoint_step_011000.pth", None),
    ("현재 최적 (full_mse, s30k)", "finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth", None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "finetune_runs/eruku_short_fullmse/_eval/gen_heldout_fonts.png"))
    ap.add_argument("--cfg", type=float, default=1.25)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-w", type=int, default=400)
    ap.add_argument("--cell-h", type=int, default=76)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    CW, CH = args.cell_w, args.cell_h

    fonts = sorted([p for p in FONTS_DIR.glob("*.otf")] + [p for p in FONTS_DIR.glob("*.ttf")])
    if not fonts:
        raise SystemExit(f"폰트 없음: {FONTS_DIR}")

    models = []
    for lbl, ck, vae in MODELS:
        print(f"loading [{lbl}] {ck}")
        m, step = load_model(ck, dev, vae_checkpoint=vae)
        models.append((lbl, m))

    col_labels = ["style ref (이 폰트)", "정답 (이 폰트)"] + [lbl for lbl, _ in models]
    ncol = len(col_labels); full_w = ncol * (CW + 4)

    def gen(model, fp, target):
        return gen_from_style(model, style_tensor(render_in_font(fp, STYLE_PHRASE)),
                              STYLE_PHRASE, target, args.cfg, args.max_new_tokens, dev,
                              seed=args.seed)

    grid = []
    hdr = header_row(col_labels, CW, CH)
    grid.append(hdr); grid.append(np.full((3, hdr.shape[1]), 80, np.uint8))
    for fp in fonts:
        writer = fp.stem.replace("AIHandwriting-", "").replace("-Hybrid-Full-Plus", "")
        for target in TARGETS:
            if not font_can_render(fp, target):
                continue
            style_ref = render_in_font(fp, STYLE_PHRASE)
            gt = render_in_font(fp, target)
            cells = [cell(style_ref, f"{writer}", CW, CH),
                     cell(gt, f"목표: {target[:22]}", CW, CH, highlight=True)]
            for lbl, m in models:
                cells.append(cell(gen(m, fp, target), lbl.split(" ")[0], CW, CH))
            row = fit_row(cells, full_w)
            grid.append(row); grid.append(np.full((6, row.shape[1]), 80, np.uint8))
            print(f"  {writer}: {target[:26]!r}")
    body = np.vstack(grid); Wd = body.shape[1]
    title = label_img("Held-out AI손글씨 폰트 4종 Eruku 생성 (학습 미사용, unseen 스타일 일반화)  "
                      f"cfg={args.cfg}   ← style | 정답 | 기존 | 현재최적 →",
                      Wd, 44, 18, bold=True, center=False, bg=255)
    out = np.vstack([title, np.full((2, Wd), 0, np.uint8), body])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
