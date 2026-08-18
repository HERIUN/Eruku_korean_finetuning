"""`style_text` 를 채웠을 때 생성물 **앞머리**에 style 잔재가 남는지 측정.

배경: style_text 가 비어있지 않으면 generate() 는 SOG 를 디코더에 넣지 않고 모델이 스스로
내야 한다. 그 사이 모델은 style 텍스트를 계속 그리므로, 크롭을 고정 오프셋으로 하면 그
echo 가 생성물 앞에 남는다. 지금 코드는 special 시퀀스의 **첫 SOG 열** 기준으로 자른다
(infer.show.crop_gen). 이 스크립트는 그 크롭 이후에도 잔재가 남는지 본다.

측정 방법 — 잔재를 사람 눈이 아니라 **리더**로 센다:
  생성물을 한글 HTR 로 읽고 목표 텍스트와 정렬해, 첫 목표 글자 **앞에 삽입된 글자 수**
  (leading insertion) 를 센다. style 잔재가 남으면 여기서 잡힌다.
  대조군으로 같은 style·목표에 `style_text=""` 를 돌린다(이 경로는 SOG 가 강제 주입돼
  echo 가 구조적으로 0). 두 조건의 차이가 곧 style_text 때문에 생긴 잔재다.

  잉크 기준도 같이 본다: 크롭 직후 앞머리 24px 의 잉크 비율(리더가 글자로 못 읽는
  얇은 획·artifact 도 잡기 위해).

사용:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/style_echo_check.py --n 60
  ... --montage _debug/style_echo.png     # 앞머리 확대 montage 도 저장
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import REPO as HERE
from infer.show import load_model, render_in_font, style_tensor, gen_from_style_batch, label_img
from eval.echo_metrics import build_texts, font_can_render
from eval.htr_cer import to_htr_input, htr_read, norm
from train.aux_htr import load_pretrained_htr
from jong_gen_loss import align


def leading_insertions(target: str, hyp: str) -> tuple[int, str]:
    """정렬에서 '첫 목표 글자 앞에 삽입된' 글자 수와 그 문자열."""
    ins = []
    for g, h in align(norm(target), norm(hyp)):
        if g is None:                 # 목표에 없는데 읽힌 글자 = 삽입
            ins.append(h or "")
            continue
        break                         # 첫 목표 글자에 도달 → 앞머리 끝
    return len(ins), "".join(ins)


def head_ink(arr: np.ndarray, px: int = 24, thr: int = 200) -> float:
    """생성물 앞머리 px 열의 잉크 비율(리더가 글자로 못 읽는 얇은 잔재도 잡는다)."""
    return float((arr[:, :px] < thr).mean()) if arr.shape[1] >= px else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth"))
    ap.add_argument("--vae-checkpoint", default=None)
    ap.add_argument("--htr-checkpoint", default=str(HERE / "finetune_runs/aux_htr_ko/htr_s20000"))
    ap.add_argument("--fonts-dir", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--max-words", type=int, default=15)
    ap.add_argument("--head-px", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--montage", default=None, help="앞머리 확대 montage 저장 경로")
    ap.add_argument("--out", default=str(HERE / "finetune_runs/_eval/style_echo.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    rng = random.Random(args.seed)
    model, step = load_model(args.ckpt, dev, vae_checkpoint=args.vae_checkpoint)
    htr = load_pretrained_htr(Path(args.htr_checkpoint)).to(dev).eval()

    # style 은 '그 폰트로 쓴 다른 문장', 목표는 별도 문장 (전사≠목표 = 실사용 조건)
    texts = build_texts(args.n * 2, rng, english=False, coherent=True, max_words=args.max_words)
    fonts = sorted(Path(args.fonts_dir).glob("*.ttf"))
    jobs = []
    for i in range(args.n):
        target, stext = texts[i], texts[args.n + i]
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, target) and font_can_render(f, stext)), None)
        if fp is not None:
            jobs.append((target, stext, render_in_font(fp, stext)))
    print(f"[setup] n={len(jobs)} (style 전사 ≠ 목표 텍스트)")

    res = {"stext": [], "nostext": []}        # 조건별 (leading_ins, ins_str, head_ink)
    heads = []                                 # montage 용 (조건, 앞머리 crop, 라벨)
    for i in range(0, len(jobs), args.batch_size):
        chunk = jobs[i:i + args.batch_size]
        sts = [style_tensor(a) for _, _, a in chunk]
        for cond, stexts in (("stext", [s for _, s, _ in chunk]), ("nostext", [""] * len(chunk))):
            gens = gen_from_style_batch(model, sts, stexts, [t for t, _, _ in chunk],
                                        args.cfg, args.max_new_tokens, dev)
            batch = torch.cat([to_htr_input(g, dev) for g in gens], 0)
            for (target, stext, _), g, hyp in zip(chunk, gens, htr_read(htr, batch, dev)):
                n_ins, ins = leading_insertions(target, hyp)
                # 앞에 붙은 글자가 **style 텍스트의 꼬리**인가? (진짜 echo 판정)
                tail = norm(stext)[-6:] if stext else ""
                echo = bool(ins) and any(c in tail for c in ins)
                res[cond].append((n_ins, ins, head_ink(g, args.head_px), echo))
                if args.montage and cond == "stext" and len(heads) < 24:
                    heads.append((g[:, :160], f"{'잔재 '+ins if ins else '깨끗'} | style끝='{stext[-6:]}'"))
        print(f"  {min(i+args.batch_size, len(jobs))}/{len(jobs)}", flush=True)

    out = {"step": step, "n": len(res["stext"]), "cfg": args.cfg, "head_px": args.head_px}
    print(f"\n=== style_text 로 인한 앞머리 잔재 (step {step}, n={len(res['stext'])}, cfg={args.cfg}) ===")
    print(f"{'조건':22s} {'앞머리이상 표본':>13s} {'그중 style꼬리':>13s} {'평균 삽입글자':>12s} {'앞머리 잉크':>12s}")
    for cond, label in (("stext", "style_text 채움"), ("nostext", "style_text=\"\" (대조군)")):
        rows = res[cond]
        rate = sum(r[0] > 0 for r in rows) / max(1, len(rows))
        echo = sum(r[3] for r in rows) / max(1, len(rows))
        mean_ins = float(np.mean([r[0] for r in rows])) if rows else 0.0
        ink = float(np.mean([r[2] for r in rows])) if rows else 0.0
        out[cond] = {"leak_rate": rate, "style_tail_rate": echo,
                     "mean_leading_ins": mean_ins, "head_ink": ink}
        print(f"{label:22s} {rate:12.3f} {echo:13.3f} {mean_ins:12.2f} {ink:12.3f}")

    leaked = Counter(r[1] for r in res["stext"] if r[1])
    if leaked:
        print("\n앞에 붙은 문자열 상위 10 (style_text 채운 조건):")
        for s, c in leaked.most_common(10):
            print(f"  {s!r}  {c}회")
    out["leading_strings"] = dict(leaked.most_common(30))

    if args.montage and heads:
        cells = [np.vstack([label_img(lab, max(320, h.shape[1]), 22, 13, center=False),
                            np.pad(h, ((0, 0), (0, max(0, 320 - h.shape[1]))), constant_values=255)])
                 for h, lab in heads]
        w = max(c.shape[1] for c in cells)
        grid = np.vstack([np.pad(c, ((0, 2), (0, w - c.shape[1])), constant_values=0) for c in cells])
        Path(args.montage).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(args.montage)
        print(f"saved {args.montage}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
