"""스타일×텍스트 매트릭스 뷰어 — 하나의 style 로 여러 gen_text 를 생성해 한눈에 비교.

infer.show 는 '행=writer, 열=cfg' 라 한 번에 한 종류 텍스트만 본다.
infer.matrix 는 '행=writer(다양한 style/폰트) × 열=다양한 gen_text' 로,
style_text·gen_text 종류를 대거 늘려 다양한 경우를 한 장에 담는다.

열 구성: [style ref] [echo(gen=style text)] [gen_text_1] [gen_text_2] ...
 - **echo** 열: gen_text 를 그 행 style 의 텍스트와 동일하게 → '본 글자 그대로 재현'(style==gen) 확인.
 - 나머지 열: 내장 다양 코퍼스(인사/가격/혼합/숫자/장문/부호/영문…) 또는 --gen-texts 로 지정.
 - 각 셀 = 모델 생성(단일 --cfg). style_text 는 각 행 ref 의 전사(--no-style-text 로 끌 수 있음).

사용:
  CUDA_VISIBLE_DEVICES=2 python infer/matrix.py --ckpt <ckpt> --n-writers 5 --cfg 1.0 \
      --lines-json data/ref_set_clean/train_lines.json --out _val/matrix.png
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))
from infer.show import load_model, load_style, gen_arr, cell, label_img

# 내장 다양 gen_text 코퍼스 (짧은/긴, 한/영/숫자/부호/혼합 골고루)
DEFAULT_GEN_TEXTS = [
    "안녕하세요 반갑습니다",                                    # 인사·단문
    "커피 한 잔에 4,500원입니다!",                              # 가격·부호
    "Hello World 한글 혼합 테스트 123",                          # 한영숫자 혼합
    "딥러닝 OCR 정확도 98.7% 달성",                              # 소수점·%
    "그는 “정말 멋지다”라고 말했다.",                   # 인용부호
    "오늘 날씨가 참 맑고 좋아서 공원으로 산책을 나갔습니다",       # 장문
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="로컬 .pth 경로 또는 HF repo id (예: HERIUN/eruku_korean)")
    ap.add_argument("--lines-json", default=str(HERE / "data" / "ref_set_clean" / "train_lines.json"))
    ap.add_argument("--out", default=str(HERE / "finetune_runs" / "matrix.png"))
    ap.add_argument("--gen-texts", nargs="+", default=None,
                    help="열로 쓸 gen_text 목록(미지정 시 내장 다양 코퍼스). echo 열은 항상 앞에 추가됨")
    ap.add_argument("--cfg", type=float, default=1.0, help="생성 CFG (매트릭스는 텍스트를 다양화하므로 cfg 는 하나 고정)")
    ap.add_argument("--n-writers", type=int, default=5, help="행 개수 = style(폰트) 수. --seed 로 추첨")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell-w", type=int, default=340)
    ap.add_argument("--cell-h", type=int, default=72)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-style-text", action="store_true",
                    help="style_text 를 빈 문자열로 (style 이미지만으로 생성). echo 열도 gen 만 style 텍스트")
    args = ap.parse_args()
    device = torch.device(args.device)
    CW, CH = args.cell_w, args.cell_h

    model, step = load_model(args.ckpt, device)

    rows_json = json.loads(Path(args.lines_json).read_text(encoding="utf-8"))
    by_w = {}
    for r in rows_json:
        by_w.setdefault(r["person_key"], []).append(r)
    rng = random.Random(args.seed)
    writers = rng.sample(list(by_w), min(args.n_writers, len(by_w)))

    gen_texts = args.gen_texts if args.gen_texts else DEFAULT_GEN_TEXTS
    tag = " (no-stext)" if args.no_style_text else ""

    grid = []
    for w in writers:
        ref = rng.choice(by_w[w])
        style_img = load_style(Path(ref["image_path"])).unsqueeze(0).to(device)
        mi = model.get_model_inputs([style_img[0]], None, style_len=style_img.shape[-1],
                                    gen_len=None, max_img_len=2048)
        dec = mi["decoder_inputs_embeds"].to(device); spx = dec.shape[1] * 8
        style_text_in = "" if args.no_style_text else ref["text"]

        ref_arr = np.array(Image.open(ref["image_path"]).convert("L"))
        cells = [cell(ref_arr, f"style ref: {w}", CW, CH)]
        # echo 열: gen == style 텍스트 (본 글자 재현)
        echo = gen_arr(model, dec, style_text_in, ref["text"], args.cfg, args.max_new_tokens, spx)
        cells.append(cell(echo, f"echo(gen=style): {ref['text'][:16]}", CW, CH, highlight=True))
        # 다양 gen_text 열
        for gt in gen_texts:
            g = gen_arr(model, dec, style_text_in, gt, args.cfg, args.max_new_tokens, spx)
            cells.append(cell(g, f"{gt[:20]}", CW, CH))
        grid.append(np.hstack(cells))
        grid.append(np.full((6, grid[-1].shape[1]), 80, np.uint8))
        print(f"  {w}: style_text={ref['text'][:30]!r}")

    body = np.vstack(grid)
    W = body.shape[1]
    title = label_img(f'스타일×텍스트 매트릭스 (step {step}, cfg={args.cfg}{tag}) | 행=style, 열=[ref|echo|다양 gen_text]',
                      W, 44, 20, bold=True, center=False, bg=255)
    # 범례: 열 gen_text 전체 나열
    legend_txt = "  |  ".join(f"{i+1}.{t}" for i, t in enumerate(gen_texts))
    legend = label_img(f'gen_text 열: {legend_txt}', W, 26, 13, center=False, bg=255)
    out = np.vstack([title, np.full((2, W), 0, np.uint8), legend, np.full((3, W), 0, np.uint8), body])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})  writers={len(writers)} cols={2+len(gen_texts)}")


if __name__ == "__main__":
    main()
