"""종성(받침) 획 손실 정량화 ① — VAE roundtrip 을 음절 단위로 픽셀 정밀 측정.

왜 VAE 부터인가: 한글 fidelity 의 하드상한은 VAE 재구성 한계다(docs/VAE_ROBUSTNESS.md).
생성물은 VAE 디코더를 반드시 통과하므로, 생성에서 보이는 획 손실 중 얼마가 VAE 탓인지
먼저 분리해야 한다. roundtrip 은 **입력과 픽셀이 정렬**되므로 음절 창을 잘라 정확히 잴 수 있다
(생성물은 정렬이 안 돼 이렇게 못 잰다 → 그건 jong_gen_loss.py 에서 HTR 로 잰다).

음절 창은 폰트 advance 로 계산한다: render_in_font 의 ink-bbox 크롭/리사이즈를 그대로
재현하면서 각 글자의 x 범위를 같이 돌려주는 render64_with_spans() 를 쓴다.

지표(각 음절 창에서 GT 잉크 기준):
  missing  = GT 잉크인데 복원에서 사라진 비율   ← '획 손실'
  extra    = GT 잉크가 아닌데 생긴 비율
  ink_keep = 복원 잉크량 / GT 잉크량
  bot_miss = 음절 잉크 bbox 의 아래 40%(=종성 자리)에서의 missing

사용:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/jong_stroke_loss.py \
      --vae finetune_runs/vae_korean_full_mse/vae_s28000 \
      --vae-base blowing-up-groundhogs/emuru_vae --n-lines 120
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL

from common import REPO as HERE, load_line_x11, roundtrip
from custom_datasets.korean.alphabet import load_charset

H = 64
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
DOUBLE = set("ㄲㅆ")                       # 쌍자음 받침(같은 자음 2개)
CLUSTER = set("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")   # 겹받침(다른 자음 2개)


def decompose(ch: str):
    """완성형 음절 → (초성, 중성, 종성). 종성 없으면 ''."""
    c = ord(ch) - 0xAC00
    return CHO[c // 588], JUNG[(c % 588) // 28], JONG[c % 28].strip()


def jong_group(j: str) -> str:
    if not j:
        return "없음"
    if j in DOUBLE:
        return "쌍받침(ㄲㅆ)"
    if j in CLUSTER:
        return "겹받침(ㄳㄵ…)"
    return "홑받침"


def render64_with_spans(font_path, text, size=72, pad=14):
    """infer.show.render_in_font + eval 의 64px 정규화를 재현하되 글자별 x 범위도 반환.

    반환: (x [1,1,64,w8] [-1,1], spans[list[(char, x0, x1)]], w64)
    """
    f = ImageFont.truetype(str(font_path), size)
    b = f.getbbox(text)
    w = max(1, b[2] - b[0]); h = max(1, b[3] - b[1])
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    ox = pad - b[0]
    ImageDraw.Draw(im).text((ox, pad - b[1]), text, font=f, fill=0)
    arr = np.array(im)
    ys, xs = np.where(arr < 250)
    if not len(ys):
        return None, [], 0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cy0, cx0 = max(0, y0 - pad), max(0, x0 - pad)
    arr = arr[cy0:y1 + pad, cx0:x1 + pad]

    # 크롭 좌표계에서의 글자 advance 경계 → 64px 정규화 배율로 스케일
    scale = H / arr.shape[0]
    adv = [f.getlength(text[:i]) for i in range(len(text) + 1)]
    spans = [(text[i], (ox + adv[i] - cx0) * scale, (ox + adv[i + 1] - cx0) * scale)
             for i in range(len(text))]

    w64 = max(1, int(round(H * arr.shape[1] / arr.shape[0])))
    return load_line_x11(arr, h=H), spans, w64      # 64px 정규화 + [-1,1] + 폭 8배수 패딩


def to01(t):
    return ((t[0, 0] + 1) / 2).numpy()


def syllable_stats(gt, rec, x0, x1, thr=0.5):
    """음절 창 [x0,x1) 의 획 손실 지표. GT 잉크가 거의 없으면 None."""
    a = gt[:, x0:x1]; b = rec[:, x0:x1]
    gi = a < thr; ri = b < thr
    n = int(gi.sum())
    if n < 20:                                    # 공백/얇은 글자 → 비율이 불안정
        return None
    miss = int((gi & ~ri).sum()) / n
    extra = int((~gi & ri).sum()) / n
    keep = int(ri.sum()) / n
    iou = int((gi & ri).sum()) / max(1, int((gi | ri).sum()))
    ys = np.where(gi.any(1))[0]                   # 이 음절 잉크의 세로 범위
    yb = ys.min() + int(0.6 * (ys.max() - ys.min() + 1))
    gb = gi[yb:]; rb = ri[yb:]
    bot = int((gb & ~rb).sum()) / max(1, int(gb.sum()))
    return dict(miss=miss, extra=extra, keep=keep, iou=iou, bot_miss=bot, ink=n)


def build_lines(n_lines, rng, per_line=12):
    """코퍼스 음절을 섞어 2~3음절 어절로 묶은 라인들. 종성 유형이 고르게 섞이도록 shuffle."""
    syl = [c for c in load_charset() if 0xAC00 <= ord(c) <= 0xD7A3]
    rng.shuffle(syl)
    lines, i = [], 0
    while len(lines) < n_lines and i + per_line <= len(syl):
        chunk = syl[i:i + per_line]; i += per_line
        words, j = [], 0
        while j < len(chunk):
            k = rng.choice([2, 3])
            words.append("".join(chunk[j:j + k])); j += k
        lines.append(" ".join(words))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=str(HERE / "finetune_runs/vae_korean_full_mse/vae_s28000"))
    ap.add_argument("--vae-base", default="blowing-up-groundhogs/emuru_vae",
                    help="비교용 원본(영어) VAE. 빈 문자열이면 생략")
    ap.add_argument("--fonts-train", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--fonts-test", default=str(HERE / "assets/fonts_korean_v2/test"))
    ap.add_argument("--n-fonts", type=int, default=8, help="train/test 각각에서 쓸 폰트 수")
    ap.add_argument("--n-lines", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "finetune_runs/_eval/jong_stroke_loss.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    rng = random.Random(args.seed)
    lines = build_lines(args.n_lines, rng)
    fonts = []
    for d, tag in ((args.fonts_train, "train"), (args.fonts_test, "test")):
        fs = sorted(Path(d).glob("*.ttf"))
        rng.shuffle(fs)
        fonts += [(f, tag) for f in fs[:args.n_fonts]]
    print(f"[setup] lines={len(lines)} fonts={len(fonts)} "
          f"({sum(t=='train' for _, t in fonts)} train / {sum(t=='test' for _, t in fonts)} test)")

    models = [("korean", args.vae)] + ([("base", args.vae_base)] if args.vae_base else [])
    vaes = [(lbl, AutoencoderKL.from_pretrained(p).to(dev).eval()) for lbl, p in models]

    acc = defaultdict(list)      # (model, tag, group) → [stats]
    per_jong = defaultdict(list)  # (model, 종성문자) → [miss]
    per_syl = defaultdict(list)   # (model, 음절) → [(miss, bot_miss, ink)]  ← H1 회귀용
    for li, text in enumerate(lines):
        for font, tag in fonts:
            x, spans, w64 = render64_with_spans(font, text)
            if x is None:                          # 잉크 0 (그 폰트가 못 그리는 글자만 있는 줄)
                continue
            gt = to01(x)
            for lbl, vae in vaes:
                rec = to01(roundtrip(vae, x, dev))
                for ch, sx0, sx1 in spans:
                    if not (0xAC00 <= ord(ch) <= 0xD7A3):
                        continue
                    a, b = int(max(0, np.floor(sx0))), int(min(w64, np.ceil(sx1)))
                    if b - a < 8:
                        continue
                    s = syllable_stats(gt, rec, a, b)
                    if s is None:
                        continue
                    j = decompose(ch)[2]
                    acc[(lbl, tag, jong_group(j))].append(s)
                    acc[(lbl, "all", jong_group(j))].append(s)
                    per_syl[(lbl, ch)].append((s["miss"], s["bot_miss"], s["ink"]))
                    if j:
                        per_jong[(lbl, j)].append(s["miss"])
        if (li + 1) % 20 == 0:
            print(f"  {li+1}/{len(lines)} lines", flush=True)

    def agg(rows, k):
        return float(np.mean([r[k] for r in rows]))

    out = {}
    print()
    for lbl, _ in models:
        for tag in ("all", "train", "test"):
            rows_any = [g for (m, t, g) in acc if m == lbl and t == tag]
            if not rows_any:
                continue
            print(f"=== VAE={lbl} 폰트={tag} ===")
            print(f"{'종성유형':16s} {'n':>6} {'missing↓':>9} {'extra':>7} {'ink_keep':>9} "
                  f"{'IoU↑':>6} {'아래40%miss↓':>12} {'GT잉크px':>8}")
            for g in ("없음", "홑받침", "쌍받침(ㄲㅆ)", "겹받침(ㄳㄵ…)"):
                rows = acc.get((lbl, tag, g))
                if not rows:
                    continue
                rec = dict(n=len(rows), miss=agg(rows, "miss"), extra=agg(rows, "extra"),
                           keep=agg(rows, "keep"), iou=agg(rows, "iou"),
                           bot=agg(rows, "bot_miss"), ink=agg(rows, "ink"))
                out[f"{lbl}|{tag}|{g}"] = rec
                print(f"{g:16s} {rec['n']:6d} {rec['miss']:9.3f} {rec['extra']:7.3f} "
                      f"{rec['keep']:9.3f} {rec['iou']:6.3f} {rec['bot']:12.3f} {rec['ink']:8.0f}")
            print()

    print("=== 종성 글자별 missing (korean VAE, n≥30, 나쁜 순 상위 15) ===")
    rows = [(j, float(np.mean(v)), len(v)) for (m, j), v in per_jong.items() if m == "korean" and len(v) >= 30]
    for j, mv, n in sorted(rows, key=lambda r: -r[1])[:15]:
        print(f"  {j}  missing={mv:.3f}  (n={n}, {jong_group(j)})")
    out["per_jong_korean"] = {j: [mv, n] for j, mv, n in rows}

    # 음절별 원자료 — H1(획 수가 진짜 설명변수인가) 회귀에 쓴다.
    out["per_syllable"] = {
        f"{lbl}|{ch}": [float(np.mean([r[0] for r in v])), float(np.mean([r[1] for r in v])),
                        float(np.mean([r[2] for r in v])), len(v)]
        for (lbl, ch), v in per_syl.items()
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
