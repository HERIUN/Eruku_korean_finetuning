"""Eruku 재적응 전/후 생성 비교 montage (같은 full_mse VAE 내장, T5 만 다름).

목적: full_mse VAE 의 복잡획 상한이 '단문강조 재적응'으로 실제 생성에서 발현되는지 눈으로 확인.
한 행 = [스타일 ref | 정답(이 폰트 렌더) | BEFORE(재적응 전 ckpt) | AFTER(현재 최적 ckpt)].
두 ckpt 는 동일 VAE 를 내장하므로 차이는 순수 T5(생성) 개선분. 복잡획 음절(밝·붉·닭·삶·값·짧…) 단문 집중.

예:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python _eval_gen_compare.py \
    --ckpt-before finetune_runs/eruku_readapt_fullvae/checkpoint_step_015000.pth \
    --ckpt-after  finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth \
    --out-dir finetune_runs/eruku_short_fullmse/_eval
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from infer_show import (load_model, load_style, render_in_font, gen_arr, cell, label_img, FONTS_DIR)
from eval_echo_metrics import font_can_render

# 복잡획(겹받침·쌍자음·밀집) 단문 집중 + 일반 단문/장문 대조군.
CATS = {
    "ko_complex_short": ["밝은 햇살", "붉은 노을", "닭볶음탕", "삶은 계란",
                         "값진 보석", "짧은 만남", "넓은 들판", "굵은 빗줄기",
                         "옳은 선택", "꿇은 무릎", "밝게 웃다", "붉게 물든"],
    "ko_short": ["한국어 평가", "인공지능 모델", "새벽 공기", "형형색색 꽃",
                 "가을 하늘", "겨울 바다", "봄날 아침", "여름 소나기"],
    "ko_long": ["다람쥐 헌 쳇바퀴에 타고파 오늘도 힘차게 달린다",
                "형형색색 꽃들이 활짝 피어난 봄날의 공원 풍경이었다",
                "밝고 붉은 노을이 넓은 들판 위로 짧게 스며들었다",
                "인공지능 모델의 한국어 생성 성능을 정밀하게 평가한다"],
    # 다양성 대상: 숫자·날짜·통화·혼합스크립트·구두점·희귀음절
    "ko_diverse": ["2024년 10월 15일", "서울 강남구 역삼동", "가격은 12,500원",
                   "AI 모델 GPT-4 평가", "온도 −3.5℃ 습도 80%", "전화 010-1234-5678",
                   "이메일: user@test.com", "쓺·뷁·꿿 희귀음절", "「한국어」 (2024)",
                   "Hello 안녕 世界 123"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-before", required=True)
    ap.add_argument("--ckpt-after", required=True)
    ap.add_argument("--label-before", default="재적응 전 (s15000)")
    ap.add_argument("--label-after", default="현재 최적 (s30000)")
    ap.add_argument("--lines-json", default=str(HERE / "data" / "ref_set_clean" / "train_lines.json"))
    ap.add_argument("--out-dir", default=str(HERE / "finetune_runs" / "eruku_short_fullmse" / "_eval"))
    ap.add_argument("--n-rows", type=int, default=6, help="카테고리별 행(스타일×텍스트) 수")
    ap.add_argument("--cfg", type=float, default=1.25)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-w", type=int, default=380)
    ap.add_argument("--cell-h", type=int, default=72)
    ap.add_argument("--exclude-writers", nargs="*", default=["UlsanJunggu"])
    ap.add_argument("--cats", nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    CW, CH = args.cell_w, args.cell_h
    outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)

    print(f"loading BEFORE {args.ckpt_before} ...")
    m_bef, step_b = load_model(args.ckpt_before, device)
    print(f"loading AFTER  {args.ckpt_after} ...")
    m_aft, step_a = load_model(args.ckpt_after, device)

    rows_all = json.loads(Path(args.lines_json).read_text(encoding="utf-8"))
    by_w = {}
    for r in rows_all:
        by_w.setdefault(r["person_key"], []).append(r)
    if args.exclude_writers:
        excl = set(args.exclude_writers)
        by_w = {w: v for w, v in by_w.items() if w.split("#")[0] not in excl}
    writers = list(by_w)
    rng = random.Random(args.seed)

    def font_of(base):
        return FONTS_DIR / f"{base}.ttf"

    def pick_writer(target):
        """target 을 렌더 가능한 폰트의 writer 를 랜덤 선택."""
        cand = writers[:]
        rng.shuffle(cand)
        for w in cand:
            fp = font_of(w.split("#")[0])
            if fp.exists() and font_can_render(fp, target):
                return w
        return None

    col_labels = ["스타일 ref (조건)", "정답 (이 폰트)", args.label_before, args.label_after]
    ncol = len(col_labels); full_w = ncol * (CW + 4)

    @torch.no_grad()
    def gen(model, ref, target):
        style_img = load_style(Path(ref["image_path"])).unsqueeze(0).to(device)
        mi = model.get_model_inputs([style_img[0]], None, style_len=style_img.shape[-1],
                                    gen_len=None, max_img_len=2048)
        dec = mi["decoder_inputs_embeds"].to(device); spx = dec.shape[1] * 8
        torch.manual_seed(args.seed)                                  # before/after 동일 조건
        return gen_arr(model, dec, ref["text"], target, args.cfg, args.max_new_tokens, spx, device)

    cats = {k: v for k, v in CATS.items() if not args.cats or k in args.cats}
    for cat, texts in cats.items():
        print(f"[{cat}]")
        grid = []
        hdr = np.hstack([cell(np.full((CH, CW), 245, np.uint8), lbl, CW, CH) for lbl in col_labels])
        grid.append(hdr); grid.append(np.full((3, hdr.shape[1]), 80, np.uint8))
        for i in range(args.n_rows):
            target = texts[i % len(texts)]
            w = pick_writer(target)
            if w is None:
                print(f"  [skip] 렌더가능 폰트 없음: {target!r}"); continue
            ref = rng.choice(by_w[w]); base = w.split("#")[0]
            fp = font_of(base)
            gt = render_in_font(fp, target)
            g_bef = gen(m_bef, ref, target)
            g_aft = gen(m_aft, ref, target)
            cells = [cell(np.array(Image.open(ref["image_path"]).convert("L")), f"ref: {base}", CW, CH),
                     cell(gt, f"목표: {target[:24]}", CW, CH, highlight=True),
                     cell(g_bef, "BEFORE", CW, CH),
                     cell(g_aft, "AFTER", CW, CH)]
            row = np.hstack(cells)
            if row.shape[1] < full_w:
                row = np.hstack([row, np.full((row.shape[0], full_w - row.shape[1]), 255, np.uint8)])
            grid.append(row); grid.append(np.full((6, row.shape[1]), 80, np.uint8))
            print(f"  {base}: {target[:30]!r}")
        body = np.vstack(grid); Wd = body.shape[1]
        title = label_img(f'Eruku 재적응 전후 생성 비교 [{cat}]  '
                          f'(BEFORE s{step_b} → AFTER s{step_a}, 동일 full_mse VAE, cfg={args.cfg})  '
                          f'← 조건 | 정답 | BEFORE | AFTER →',
                          Wd, 44, 18, bold=True, center=False, bg=255)
        out = np.vstack([title, np.full((2, Wd), 0, np.uint8), body])
        p = outd / f"gencmp_{cat}.png"
        Image.fromarray(out).save(p)
        print(f"  saved {p}  ({out.shape[1]}x{out.shape[0]})")
    print("done.")


if __name__ == "__main__":
    main()
