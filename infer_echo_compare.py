"""echo(style==gen) 모델 비교 뷰어 — 같은 영어 텍스트를 GT 옆에 각 모델 생성물로 나란히.

행 = 샘플(다양 길이 영어), 열 = [GT(정답 렌더) | pretrained | OLD20k | v2_20k | v2_30k].
각 모델은 GT 를 style 로 주고 같은 텍스트를 echo 생성. 픽셀 vs 지각 차이를 눈으로 확인.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from infer_show import load_model, render_in_font, style_tensor, gen_from_style, cell, label_img
from eval_echo_metrics import font_can_render as can_render

PRE = "/home/boinit/.cache/huggingface/hub/models--blowing-up-groundhogs--eruku/snapshots/13f4f7501f0f1d7270da08fded134c57d3635705/pytorch_model.bin"
MODELS = [
    ("pretrained", PRE),
    ("OLD 20k",    "finetune_runs/korean_en2ko/checkpoint_step_020000.pth"),
    ("v2 20k",     "finetune_runs/korean_en2ko_v2/checkpoint_step_020000.pth"),
    ("v2 30k",     "finetune_runs/korean_en2ko_v2/checkpoint_step_030000.pth"),
]
TEXTS = [
    "The quick brown fox jumps over the lazy dog 2024",
    "In the beginning there was nothing but darkness and then a brilliant light appeared",
    "Machine learning models can generate handwritten text in many different styles and fonts today",
    "She sells seashells by the seashore while the waves crash gently against the rocks at dawn",
    "The five boxing wizards jump quickly over the fence as the storm approaches from the west 1234",
]
FONTS_DIR = HERE / "assets" / "fonts_korean_v2" / "train"
CW, CH = 380, 72


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-style-text", action="store_true", help="style_text 를 빈 문자열로(style 이미지만)")
    ap.add_argument("--out", default=str(HERE / "finetune_runs/korean_en2ko_v2/_val/echo_compare_eng.png"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fonts = sorted(FONTS_DIR.glob("*.ttf"))
    samples = []
    for i, t in enumerate(TEXTS):
        fp = next((f for f in fonts[i::7] if can_render(f, t)), None) or next(f for f in fonts if can_render(f, t))
        gt = render_in_font(fp, t)
        samples.append((t, fp.stem, gt))

    gens = {name: [] for name, _ in MODELS}
    for name, ck in MODELS:
        print(f"loading {name} ...")
        model, step = load_model(ck, device)
        for t, _, gt in samples:
            stext = "" if args.no_style_text else t
            g = gen_from_style(model, style_tensor(gt), stext, t, 1.0, 480, device)  # max_new 480 (긴문장)
            gens[name].append(g)
        del model
        torch.cuda.empty_cache()

    rows = []
    for si, (t, fname, gt) in enumerate(samples):
        cells = [cell(gt, f"GT[{fname}]: {t[:22]}", CW, CH, highlight=True)]
        for name, _ in MODELS:
            cells.append(cell(gens[name][si], f"{name}", CW, CH))
        rows.append(np.hstack(cells))
        rows.append(np.full((6, rows[-1].shape[1]), 80, np.uint8))
    body = np.vstack(rows)
    W = body.shape[1]
    tag = "style_text 없음" if args.no_style_text else "style_text=목표텍스트(echo)"
    title = label_img(f"echo 비교 (영어 긴문장, cfg=1.0, {tag}) | 열=[GT | pretrained | OLD20k | v2_20k | v2_30k]",
                      W, 40, 18, bold=True, center=False, bg=255)
    out = np.vstack([title, np.full((3, W), 0, np.uint8), body])
    op = Path(args.out); op.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(op)
    print(f"saved {op}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
