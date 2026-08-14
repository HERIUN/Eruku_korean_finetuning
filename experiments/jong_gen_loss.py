"""종성(받침) 획 손실 정량화 ② — 생성물에서 자모 단위로 측정.

생성 이미지는 GT 와 픽셀 정렬이 안 되므로(폭·자간이 다름) VAE roundtrip 때처럼 창을 잘라
잴 수 없다. 대신 한글 HTR 리더로 읽어 **음절 정렬 후 자모 분해**해서 센다:
받침이 획 손실로 뭉개지면 리더는 그 음절을 받침 없는 음절로 읽는다(예: 한→하) → 종성 삭제율.

세 경로를 같은 지표로 재서 손실을 분해한다:
  GT 렌더        : 리더 자체 오차(바닥)
  GT → VAE 복원  : VAE 가 만드는 상한 손실
  생성           : 최종 손실 (T5 + VAE)

사용:
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/jong_gen_loss.py \
      --ckpt finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth --n 200
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common import REPO as HERE, load_line_x11, roundtrip, to_u8
from infer.show import load_model, render_in_font, style_tensor, gen_from_style_batch
from eval.echo_metrics import build_texts, font_can_render
from eval.htr_cer import to_htr_input, htr_read, norm
from train.aux_htr import load_pretrained_htr
from jong_stroke_loss import decompose, jong_group  # noqa: E402
from custom_datasets.korean_alphabet import load_charset  # noqa: E402

GROUPS = ("없음", "홑받침", "쌍받침(ㄲㅆ)", "겹받침(ㄳㄵ…)")


def build_balanced(n_lines, rng, per_group=3):
    """종성 유형이 균등하게 섞인 라인들.

    실제 한국어에서 쌍받침/겹받침은 드물어(코헤런트 텍스트 n=200 기준 음절의 2~3%)
    자연 분포만으론 그 칸의 표본이 부족하다. 유형별 표본을 맞춘 통제 비교용.
    (랜덤 음절 나열이라 텍스트 자체는 OOD — 유형 간 '차이' 를 보는 용도지
     절대 수준을 코헤런트 결과와 섞어 비교하면 안 된다.)
    """
    by = {g: [] for g in GROUPS}
    for c in load_charset():
        if 0xAC00 <= ord(c) <= 0xD7A3:
            by[jong_group(decompose(c)[2])].append(c)
    lines = []
    for _ in range(n_lines):
        chars = [rng.choice(by[g]) for g in GROUPS for _ in range(per_group)]
        rng.shuffle(chars)
        words, j = [], 0
        while j < len(chars):
            k = rng.choice([2, 3]); words.append("".join(chars[j:j + k])); j += k
        lines.append(" ".join(words))
    return lines


def build_natural_jong(n_lines, rng, per_line_cluster=2, per_line_other=2):
    """겹받침을 포함한 **실제 단어**로 라인을 구성 (M0 벤치마크).

    balanced 모드는 종성 유형 표본을 맞추려고 코퍼스 음절을 무작위 나열해서 텍스트 자체가 OOD 다.
    반대로 자연 문장(coherent)은 겹받침 음절이 0.4% 뿐이라 그 칸을 못 잰다.
    그 사이를 메운다 — **어휘는 실제 단어 그대로 두고 겹받침 포함 단어만 골라** 표본을 키운다.
    한 라인에 겹받침 단어와 일반 단어를 섞으므로 대조군(홑받침/없음)이 같은 이미지에서 나온다.
    """
    import gen_korean_fontset as G
    A = HERE / "assets"
    ko = G.build_pools(A / "corpus/korean_lines.txt", A / "corpus/chars.txt",
                       str(A / "corpus/english_words.txt"), 0, rng)["ko"]
    def has_cluster(w):
        return any(0xAC00 <= ord(c) <= 0xD7A3 and jong_group(decompose(c)[2]) == "겹받침(ㄳㄵ…)"
                   for c in w)
    cl = [w for w in ko if has_cluster(w)]
    other = [w for w in ko if not has_cluster(w)]
    print(f"[M0] 겹받침 포함 실제 단어 {len(cl):,} / 그 외 {len(other):,}")
    lines = []
    for _ in range(n_lines):
        toks = ([rng.choice(cl) for _ in range(per_line_cluster)] +
                [rng.choice(other) for _ in range(per_line_other)])
        rng.shuffle(toks)
        lines.append(" ".join(toks))
    return lines


def align(a: str, b: str):
    """문자 단위 Needleman-Wunsch(단위 비용) 정렬 → [(a_ch|None, b_ch|None)]."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            out.append((a[i - 1], None)); i -= 1
        else:
            out.append((None, b[j - 1])); j -= 1
    return out[::-1]


def is_syl(c):
    return c is not None and 0xAC00 <= ord(c) <= 0xD7A3


def tally(gt_text, hyp, acc, tag, per_syl=None):
    """정렬 후 GT 음절별로 종성 처리 결과를 센다. per_syl 주면 음절별 정확도도 누적(H2 회귀용)."""
    for g, h in align(norm(gt_text), norm(hyp)):
        if not is_syl(g):
            continue
        cg, vg, jg = decompose(g)
        grp = jong_group(jg)
        r = acc[(tag, grp)]
        r["n"] += 1
        if per_syl is not None:
            s = per_syl[g]
            s["n"] += 1
            s["exact"] += int(g == h)
            s["jong_ok"] += int(is_syl(h) and decompose(h)[2] == jg)
        if not is_syl(h):                       # 음절이 통째로 사라지거나 비한글로 읽힘
            r["syl_del"] += 1
            if jg:
                r["jong_del"] += 1              # 종성도 같이 소실 (보수적으로 삭제로 셈)
            continue
        ch, vh, jh = decompose(h)
        r["exact"] += int(g == h)
        r["cho_err"] += int(ch != cg)
        r["jung_err"] += int(vh != vg)
        if jg and not jh:
            r["jong_del"] += 1
        elif jg and jh != jg:
            r["jong_sub"] += 1
        elif not jg and jh:
            r["jong_ins"] += 1


def vae_recon_arr(model, arr, dev):
    """GT 렌더 → 모델의 VAE roundtrip → uint8 라인이미지 (생성과 같은 디코더 경로)."""
    w64 = max(1, int(round(64 * arr.shape[1] / arr.shape[0])))
    return to_u8(roundtrip(model.vae, load_line_x11(arr), dev)[..., :w64])   # 8배수 패딩분 제거


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth"))
    ap.add_argument("--vae-checkpoint", default=None)
    ap.add_argument("--htr-checkpoint", default=str(HERE / "finetune_runs/aux_htr_ko/htr_s20000"))
    ap.add_argument("--fonts-dir", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--texts", choices=["coherent", "balanced", "natural_jong"], default="coherent",
                    help="coherent=실제 문장(자연 분포) / balanced=종성 유형 균등(통제, OOD) / "
                         "natural_jong=겹받침 포함 실제 단어(M0: 자연 어휘로 겹받침 표본 확보)")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="generate_batch 배치 크기 (KV 캐시 + 배치로 샘플당 비용↓)")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--max-words", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "finetune_runs/_eval/jong_gen_loss.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    rng = random.Random(args.seed)
    model, step = load_model(args.ckpt, dev, vae_checkpoint=args.vae_checkpoint)
    htr = load_pretrained_htr(Path(args.htr_checkpoint)).to(dev).eval()
    if args.texts == "balanced":
        texts = build_balanced(args.n, rng)
    elif args.texts == "natural_jong":
        texts = build_natural_jong(args.n, rng)
    else:
        texts = build_texts(args.n, rng, english=False, coherent=True, max_words=args.max_words)
    fonts = sorted(Path(args.fonts_dir).glob("*.ttf"))

    acc = defaultdict(lambda: defaultdict(int))
    per_syl = defaultdict(lambda: defaultdict(int))   # 생성 경로 음절별 정확도 (H2 회귀용)

    # 폰트 배정을 먼저 끝내고 배치로 묶는다. generate_batch 의 스텝 수는 '가장 늦게 EOG 를
    # 내는 샘플'이 정하므로, 길이가 비슷한 것끼리 묶어야 낭비가 적다 → 텍스트 길이로 정렬.
    jobs = []
    for text in texts:
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text)), None)
        if fp is not None:
            jobs.append((text, render_in_font(fp, text)))
    jobs.sort(key=lambda t: len(t[0]))

    done = 0
    for i in range(0, len(jobs), args.batch_size):
        chunk = jobs[i:i + args.batch_size]
        sts = [style_tensor(gt) for _, gt in chunk]
        gens = gen_from_style_batch(model, sts, [t for t, _ in chunk], [t for t, _ in chunk],
                                    args.cfg, args.max_new_tokens, dev)
        recs = [vae_recon_arr(model, gt, dev) for _, gt in chunk]
        # HTR 리더도 최대 150 스텝 자기회귀다 → 낱개로 부르면 여기가 병목이 된다.
        # to_htr_input 이 폭 768 로 고정해 주므로 그대로 배치로 묶어 한 번에 읽는다.
        for tag, imgs in (("GT렌더(리더바닥)", [gt for _, gt in chunk]),
                          ("VAE복원", recs), ("생성", gens)):
            batch = torch.cat([to_htr_input(im, dev) for im in imgs], 0)
            for (text, _), hyp in zip(chunk, htr_read(htr, batch, dev)):
                tally(text, hyp, acc, tag, per_syl=per_syl if tag == "생성" else None)
        done += len(chunk)
        print(f"  {done}/{len(jobs)}", flush=True)

    out = {}
    for tag in ("GT렌더(리더바닥)", "VAE복원", "생성"):
        print(f"\n=== {tag} (step {step}, n_text={done}, cfg={args.cfg}, texts={args.texts}) ===")
        print(f"{'종성유형':16s} {'음절수':>7} {'음절정확도↑':>11} {'종성삭제율↓':>11} "
              f"{'종성오치환↓':>11} {'초성오류':>9} {'중성오류':>9}")
        for grp in ("없음", "홑받침", "쌍받침(ㄲㅆ)", "겹받침(ㄳㄵ…)"):
            r = acc.get((tag, grp))
            if not r or not r["n"]:
                continue
            n = r["n"]
            rec = dict(n=n, exact=r["exact"] / n, jong_del=r["jong_del"] / n,
                       jong_sub=r["jong_sub"] / n, jong_ins=r["jong_ins"] / n,
                       cho_err=r["cho_err"] / n, jung_err=r["jung_err"] / n,
                       syl_del=r["syl_del"] / n)
            out[f"{tag}|{grp}"] = rec
            print(f"{grp:16s} {n:7d} {rec['exact']:11.3f} {rec['jong_del']:11.3f} "
                  f"{rec['jong_sub']:11.3f} {rec['cho_err']:9.3f} {rec['jung_err']:9.3f}")
        r0 = acc.get((tag, "없음"))
        if r0 and r0["n"]:
            print(f"{'(참고) 종성없음 음절에 종성이 생긴 비율':46s} {r0['jong_ins']/r0['n']:.3f}")

    # 음절별 생성 정확도 — 학습 빈도/획 수와의 상관 분석(jong_hypotheses.py)에 쓴다
    out["per_syllable"] = {ch: [v["exact"] / v["n"], v["jong_ok"] / v["n"], v["n"]]
                           for ch, v in per_syl.items() if v["n"]}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
