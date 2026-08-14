"""VAE 교체 전/후(old vs 한글적응 new) 생성 비교 montage.

decoder-only VAE 는 encoder/T5 가 동일 → 같은 style·같은 예측 latent 에서 **decode 만 다름**.
따라서 한 행에 [스타일 ref | 정답 | OLD 생성 | NEW 생성] 을 나란히 놓으면 VAE 효과만 격리해 보인다.

카테고리: 한글 단문/장문, 영어 단문/장문. 카테고리별 montage PNG 저장.

예:
  CUDA_VISIBLE_DEVICES=2 python vae_compare.py \
    --ckpt finetune_runs/korean_en2ko_fixed/checkpoint_step_011000.pth \
    --vae-checkpoint finetune_runs/vae_korean_dec/vae_s15000 \
    --out-dir finetune_runs/korean_en2ko_fixed/_eval
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from PIL import Image

from common import REPO, header_row, fit_row
from infer.show import (load_model, load_style, render_in_font, gen_from_style,
                        cell, label_img, FONTS_DIR)

CATS = {
    "ko_short": ["한국어 평가", "형형색색 꽃", "인공지능 모델", "새벽 공기"],
    "ko_long": ["다람쥐 헌 쳇바퀴에 타고파 오늘도 힘차게 달린다",
                "인공지능 모델의 한국어 생성 성능을 정밀하게 평가한다",
                "형형색색 꽃들이 활짝 피어난 봄날의 공원 풍경이었다",
                "빠른 갈색 여우가 게으른 개를 뛰어넘어 강가로 갔다"],
    "en_short": ["Hello World", "Model Test", "Quick Fox", "Spring Day"],
    "en_long": ["The quick brown fox jumps over the lazy dog by the river",
                "Evaluate the Korean generation quality of this model today",
                "Colorful flowers bloom across the sunny spring park now",
                "Autoregressive text image generation with a finetuned decoder"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae-checkpoint", required=True, help="한글적응 new VAE(vae_sXXXX)")
    ap.add_argument("--lines-json", default=str(REPO / "data" / "korean_fontset_v2" / "train_lines.json"))
    ap.add_argument("--out-dir", default=str(REPO / "finetune_runs" / "korean_en2ko_fixed" / "_eval"))
    ap.add_argument("--n-writers", type=int, default=4, help="행(스타일) 수 = 카테고리 텍스트 수만큼")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-w", type=int, default=380)
    ap.add_argument("--cell-h", type=int, default=72)
    ap.add_argument("--exclude-writers", nargs="*", default=["UlsanJunggu"])
    ap.add_argument("--cats", nargs="*", default=None, help="생성할 카테고리 필터(기본 전체)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    CW, CH = args.cell_w, args.cell_h
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)

    print("loading OLD (원본 VAE)...")
    m_old, step = load_model(args.ckpt, device)                       # 옛 vae
    print("loading NEW (한글적응 VAE)...")
    m_new, _ = load_model(args.ckpt, device, vae_checkpoint=args.vae_checkpoint)

    rows_all = json.loads(Path(args.lines_json).read_text(encoding="utf-8"))
    by_w = {}
    for r in rows_all:
        by_w.setdefault(r["person_key"], []).append(r)
    if args.exclude_writers:
        excl = set(args.exclude_writers)
        by_w = {w: v for w, v in by_w.items() if w.split("#")[0] not in excl}
    rng = random.Random(args.seed)
    writers = rng.sample(list(by_w), min(args.n_writers, len(by_w)))   # 카테고리 간 동일 스타일

    col_labels = ["스타일 ref (조건)", "정답 (이 폰트)", f"OLD VAE 생성 (cfg={args.cfg})",
                  f"NEW VAE 생성 (cfg={args.cfg})"]
    ncol = len(col_labels); full_w = ncol * (CW + 4)

    def gen(model, ref, target):
        return gen_from_style(model, load_style(Path(ref["image_path"])), ref["text"], target,
                              args.cfg, args.max_new_tokens, device,
                              seed=args.seed)                         # old/new 동일 조건

    cats = {k: v for k, v in CATS.items() if not args.cats or k in args.cats}
    for cat, texts in cats.items():
        print(f"[{cat}]")
        grid = []
        hdr = header_row(col_labels, CW, CH)
        grid.append(hdr); grid.append(np.full((3, hdr.shape[1]), 80, np.uint8))
        for i, w in enumerate(writers):
            ref = rng.choice(by_w[w])
            target = texts[i % len(texts)]
            base = w.split("#")[0]
            fp = Path(ref["font_path"]) if ref.get("font_path") else FONTS_DIR / f"{base}.ttf"
            gt = render_in_font(fp, target) if fp.exists() else np.full((64, 200), 230, np.uint8)
            g_old = gen(m_old, ref, target)
            g_new = gen(m_new, ref, target)
            cells = [cell(np.array(Image.open(ref["image_path"]).convert("L")), f"ref: {base}", CW, CH),
                     cell(gt, f"목표: {target[:24]}", CW, CH, highlight=True),
                     cell(g_old, "OLD", CW, CH),
                     cell(g_new, "NEW", CW, CH)]
            row = fit_row(cells, full_w)
            grid.append(row); grid.append(np.full((6, row.shape[1]), 80, np.uint8))
            print(f"  {base}: {target[:30]!r}")
        body = np.vstack(grid); Wd = body.shape[1]
        title = label_img(f'VAE 교체 전후 비교 [{cat}]  (step {step}, cfg={args.cfg})  '
                          f'← 조건 | 정답모양 | OLD 생성 | NEW 생성 →',
                          Wd, 44, 20, bold=True, center=False, bg=255)
        out = np.vstack([title, np.full((2, Wd), 0, np.uint8), body])
        p = outd / f"compare_{cat}.png"
        Image.fromarray(out).save(p)
        print(f"  saved {p}  ({out.shape[1]}x{out.shape[0]})")
    print("done.")


if __name__ == "__main__":
    main()
