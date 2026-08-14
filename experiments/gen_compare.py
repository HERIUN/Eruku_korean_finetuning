"""Eruku 두 모델의 생성 비교 montage.

한 행 = [스타일 ref | 정답(이 폰트 렌더) | BEFORE | AFTER]. `--vae-before/--vae-after` 를 주면
그 VAE 로 교체해(ckpt 의 옛 vae.* strip) 생성하므로, T5 만 다르게도 / VAE 까지 다르게도 비교 가능.

예 1) 재적응 전후 (동일 full_mse VAE 내장 → 차이는 순수 T5 개선분):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python gen_compare.py \
    --ckpt-before finetune_runs/eruku_readapt_fullvae/checkpoint_step_015000.pth \
    --ckpt-after  finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth \
    --out-dir finetune_runs/eruku_short_fullmse/_eval

예 2) 릴리즈 figure — 원본 영어 pretrained(원본 VAE) vs 릴리즈(한글 VAE):
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python gen_compare.py \
    --ckpt-before ~/.cache/huggingface/hub/models--blowing-up-groundhogs--eruku/snapshots/*/pytorch_model.bin \
    --ckpt-after  finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth \
    --label-before "원본 pretrained (영어, 원본 VAE)" --label-after "릴리즈 (한글 VAE)" \
    --cats ko_release en_release --n-rows 4 --out-dir /tmp/relfig
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from PIL import Image

from common import REPO, header_row, fit_row
from infer.show import (load_model, render_in_font, style_tensor, gen_from_style,
                        cell, label_img, FONTS_DIR)
from eval.echo_metrics import font_can_render

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
    # 릴리즈 카드용: 단문/장문 섞어 한 장에 (한글)
    "ko_release": ["한국어 손글씨 생성", "밝은 햇살 붉은 노을",
                   "형형색색 꽃들이 활짝 피어난 봄날의 공원",
                   "2024년 10월 15일 서울특별시 강남구"],
    # 릴리즈 카드용: 영어 능력 보존 확인
    "en_release": ["Hello world", "handwriting generation",
                   "The quick brown fox jumps over the lazy dog",
                   "Autoregressive styled text image generation"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-before", required=True)
    ap.add_argument("--ckpt-after", required=True)
    ap.add_argument("--label-before", default="재적응 전 (s15000)")
    ap.add_argument("--label-after", default="현재 최적 (s30000)")
    ap.add_argument("--vae-before", default=None,
                   help="BEFORE 모델에 물릴 VAE(vae_sXXXX 또는 hub id). 주면 ckpt 의 옛 vae.* strip")
    ap.add_argument("--vae-after", default=None, help="AFTER 모델에 물릴 VAE (같은 규칙)")
    ap.add_argument("--title", default=None, help="montage 제목(미지정 시 자동)")
    ap.add_argument("--col-labels", nargs=2, default=["스타일 ref (조건)", "정답 (이 폰트)"],
                    help="1·2열 라벨(조건/정답). 공개 figure 는 영문으로 넘긴다")
    ap.add_argument("--row-label-fmt", nargs=2, default=["ref: {font}", "목표: {text}"],
                    help="1·2열 셀 라벨 포맷({font}/{text} 치환)")
    ap.add_argument("--lines-json", default=str(REPO / "data" / "ref_set_clean" / "train_lines.json"))
    ap.add_argument("--style-fonts-dir", default=None,
                    help="주면 lines_json 대신 이 디렉토리 폰트로 style ref 를 즉석 렌더(--style-text 를 씀). "
                         "영어 pretrained 와 비교할 때 필수 — 한글폰트 style 은 그 모델엔 OOD 라 "
                         "타깃 스크립트와 무관하게 붕괴해서 baseline 을 과소평가한다")
    ap.add_argument("--style-text", default="The quick brown fox jumps",
                    help="--style-fonts-dir 사용 시 style ref 로 렌더할 텍스트(전사로도 조건에 들어감)")
    ap.add_argument("--out-dir", default=str(REPO / "finetune_runs" / "eruku_short_fullmse" / "_eval"))
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
    m_bef, step_b = load_model(args.ckpt_before, device, vae_checkpoint=args.vae_before)
    print(f"loading AFTER  {args.ckpt_after} ...")
    m_aft, step_a = load_model(args.ckpt_after, device, vae_checkpoint=args.vae_after)

    rng = random.Random(args.seed)
    render_style = bool(args.style_fonts_dir)      # style ref 를 즉석 렌더할지

    if render_style:
        style_fonts = sorted(p for p in Path(args.style_fonts_dir).glob("*")
                             if p.suffix.lower() in (".ttf", ".otf"))
        by_w, writers = {}, [p.stem for p in style_fonts]
        _fonts_by_stem = {p.stem: p for p in style_fonts}
        print(f"style ref: {len(style_fonts)}개 폰트에서 즉석 렌더 (text={args.style_text!r})")
    else:
        rows_all = json.loads(Path(args.lines_json).read_text(encoding="utf-8"))
        by_w = {}
        for r in rows_all:
            by_w.setdefault(r["person_key"], []).append(r)
        if args.exclude_writers:
            excl = set(args.exclude_writers)
            by_w = {w: v for w, v in by_w.items() if w.split("#")[0] not in excl}
        writers = list(by_w)

    def font_of(base):
        return _fonts_by_stem[base] if render_style else FONTS_DIR / f"{base}.ttf"

    def pick_writer(target):
        """target(+즉석 렌더 시 style 텍스트)을 렌더 가능한 폰트의 writer 를 랜덤 선택."""
        cand = writers[:]
        rng.shuffle(cand)
        for w in cand:
            fp = font_of(w.split("#")[0])
            if not fp.exists() or not font_can_render(fp, target):
                continue
            if render_style and not font_can_render(fp, args.style_text):
                continue
            return w
        return None

    def make_ref(w):
        """writer 키 w → style ref {text, arr}. 즉석 렌더 모드면 그 폰트로 style 텍스트를 렌더."""
        if render_style:
            return {"text": args.style_text,
                    "arr": render_in_font(font_of(w.split("#")[0]), args.style_text)}
        r = rng.choice(by_w[w])
        return {"text": r["text"], "arr": np.array(Image.open(r["image_path"]).convert("L"))}

    col_labels = [*args.col_labels, args.label_before, args.label_after]
    ncol = len(col_labels); full_w = ncol * (CW + 4)

    def gen(model, ref, target):
        return gen_from_style(model, style_tensor(ref["arr"]), ref["text"], target,
                              args.cfg, args.max_new_tokens, device,
                              seed=args.seed)                         # before/after 동일 조건

    cats = {k: v for k, v in CATS.items() if not args.cats or k in args.cats}
    for cat, texts in cats.items():
        print(f"[{cat}]")
        grid = []
        hdr = header_row(col_labels, CW, CH)
        grid.append(hdr); grid.append(np.full((3, hdr.shape[1]), 80, np.uint8))
        for i in range(args.n_rows):
            target = texts[i % len(texts)]
            w = pick_writer(target)
            if w is None:
                print(f"  [skip] 렌더가능 폰트 없음: {target!r}"); continue
            base = w.split("#")[0]; ref = make_ref(w)
            fp = font_of(base)
            gt = render_in_font(fp, target)
            g_bef = gen(m_bef, ref, target)
            g_aft = gen(m_aft, ref, target)
            cells = [cell(ref["arr"],
                          args.row_label_fmt[0].format(font=base, text=target[:24]), CW, CH),
                     cell(gt, args.row_label_fmt[1].format(font=base, text=target[:24]), CW, CH, highlight=True),
                     cell(g_bef, "BEFORE", CW, CH),
                     cell(g_aft, "AFTER", CW, CH)]
            row = fit_row(cells, full_w)
            grid.append(row); grid.append(np.full((6, row.shape[1]), 80, np.uint8))
            print(f"  {base}: {target[:30]!r}")
        body = np.vstack(grid); Wd = body.shape[1]
        # 기본 제목은 영문 — 이 figure 는 HF 모델카드로 그대로 발행된다
        auto = (f'Eruku generation [{cat}] — BEFORE step {step_b} vs AFTER step {step_a}, '
                f'cfg={args.cfg}   (columns: input | target | before | after)')
        title = label_img(auto if args.title is None else args.title,
                          Wd, 44, 18, bold=True, center=False, bg=255)
        out = np.vstack([title, np.full((2, Wd), 0, np.uint8), body])
        p = outd / f"gencmp_{cat}.png"
        Image.fromarray(out).save(p)
        print(f"  saved {p}  ({out.shape[1]}x{out.shape[0]})")
    print("done.")


if __name__ == "__main__":
    main()
