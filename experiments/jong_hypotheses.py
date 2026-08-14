"""종성 획 손실 가설 검증 — H1(획 수) / H2(학습 빈도) 재분석.

기존 측정 산출물만 재분석한다(GPU 불필요, 새 생성 없음):
  --vae-json  : jong_stroke_loss.py 의 per_syllable 덤프  → H1
  --gen-json  : jong_gen_loss.py 의 per_syllable 덤프     → H2b

H1  획 수가 진짜 설명변수인가?
    "겹받침이라서 나쁘다" vs "획이 많아서 나쁘다" 를 가른다. 핵심은 **획 수를 통제한 뒤에도
    겹받침이 홑받침보다 나쁜가** — 나쁘지 않다면 원인은 받침 종류가 아니라 획 밀도다.

H2  학습 빈도가 설명변수인가?
    코퍼스(assets/corpus)에서 음절/종성 빈도를 세고, 생성 정확도와의 상관을 본다.
    빈도가 원인이면 희귀 음절일수록 나쁘고, 빈도를 통제하면 겹받침 효과가 사라져야 한다.

사용:
  .venv/bin/python experiments/jong_hypotheses.py --vae-json <...> [--gen-json <...>]
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from common import REPO as HERE
from jong_stroke_loss import decompose, jong_group

CORPUS = HERE / "assets" / "corpus"

# 자모별 획 수(표준 필순 기준). 쌍자음은 2배, 겹받침/복합모음은 구성 자모의 합.
STROKE = {
    "ㄱ": 1, "ㄴ": 1, "ㄷ": 2, "ㄹ": 3, "ㅁ": 3, "ㅂ": 4, "ㅅ": 2, "ㅇ": 1,
    "ㅈ": 2, "ㅊ": 3, "ㅋ": 2, "ㅌ": 3, "ㅍ": 4, "ㅎ": 3,
    "ㄲ": 2, "ㄸ": 4, "ㅃ": 8, "ㅆ": 4, "ㅉ": 4,
    "ㅏ": 2, "ㅑ": 3, "ㅓ": 2, "ㅕ": 3, "ㅗ": 2, "ㅛ": 3, "ㅜ": 2, "ㅠ": 3,
    "ㅡ": 1, "ㅣ": 1, "ㅐ": 3, "ㅒ": 4, "ㅔ": 3, "ㅖ": 4,
    "ㅘ": 4, "ㅙ": 5, "ㅚ": 3, "ㅝ": 4, "ㅞ": 5, "ㅟ": 3, "ㅢ": 2,
    "ㄳ": 3, "ㄵ": 3, "ㄶ": 4, "ㄺ": 4, "ㄻ": 6, "ㄼ": 7, "ㄽ": 5, "ㄾ": 6,
    "ㄿ": 7, "ㅀ": 6, "ㅄ": 6,
}
# 겹받침의 구성 자모(자모 '개수' 를 세기 위해)
CLUSTER_PARTS = {"ㄳ": 2, "ㄵ": 2, "ㄶ": 2, "ㄺ": 2, "ㄻ": 2, "ㄼ": 2, "ㄽ": 2,
                 "ㄾ": 2, "ㄿ": 2, "ㅀ": 2, "ㅄ": 2}


def strokes(ch: str) -> int:
    c, v, j = decompose(ch)
    return STROKE.get(c, 0) + STROKE.get(v, 0) + (STROKE.get(j, 0) if j else 0)


def n_jamo(ch: str) -> int:
    """자모 '개수'(겹받침은 2로 셈) — 획 수와 구분되는 별개 설명변수."""
    c, v, j = decompose(ch)
    return 2 + (CLUSTER_PARTS.get(j, 1) if j else 0)


def corpus_counts():
    """코퍼스 음절 빈도 Counter. 한글 음절만.

    ⚠️ 이건 **학습 노출량이 아니다**. 학습 샘플러(`MixedLineSampler`)는 코퍼스 토큰 빈도를 쓰지 않고
    `build_pools` 가 만든 **고유 단어 집합에서 균등 추출**하며, 별도 `rand` 카테고리가 음절을 균등
    노출한다. 노출 기반 분석은 아래 train_exposure() 를 쓸 것.
    """
    cnt = Counter()
    for name in ("korean_lines.txt", "chars.txt"):
        fp = CORPUS / name
        if fp.exists():
            cnt.update(c for c in fp.read_text(encoding="utf-8") if 0xAC00 <= ord(c) <= 0xD7A3)
    return cnt


def train_exposure(seed=42):
    """학습 샘플러가 실제로 음절을 노출하는 기대량 (토큰 1개당 기대 등장 수).

    `custom_datasets.korean_split.build_samplers` 의 구성을 그대로 재현해 해석적으로 계산한다:
      - 가중치 ko 0.45 / en 0.20 / num 0.20 / rand 0.15
      - ko 토큰 = `build_pools` 의 **고유** 한글 단어에서 균등 추출 (코퍼스 빈도 아님!)
      - rand 토큰 = 전 폰트 교집합 음절(KS X 1001 2350자)에서 균등, 길이 1~4(평균 2.55)
    → 노출 = 0.45 · (단어풀에서 그 음절이 나오는 총 횟수 / 단어 수) + 0.15 · 2.55 / |rand풀|
    """
    import json as _json, random as _random
    import custom_datasets.korean_fontset as G
    A = HERE / "assets"
    pools = G.build_pools(A / "corpus/korean_lines.txt", A / "corpus/chars.txt",
                          str(A / "corpus/english_words.txt"), 8000, _random.Random(seed))
    inter = None
    for s in _json.load(open(A / "fonts_korean_v2/train/fonts_charsets.json")).values():
        f = {ord(c) for c in s}
        inter = f if inter is None else inter & f
    syls = {chr(c) for c in inter if 0xAC00 <= c <= 0xD7A3}
    ko = pools["ko"]
    ko_cnt = Counter(c for w in ko for c in w if 0xAC00 <= ord(c) <= 0xD7A3)
    mean_n = 1 * 0.15 + 2 * 0.35 + 3 * 0.30 + 4 * 0.20
    floor = 0.15 * mean_n / len(syls)
    exp = {}
    for c in set(ko_cnt) | syls:
        exp[c] = 0.45 * ko_cnt.get(c, 0) / len(ko) + (floor if c in syls else 0.0)
    return exp


def ols_std(y, X, names):
    """표준화 다중회귀 — 교란된 설명변수(획 수 vs 빈도 vs 겹받침)를 서로 통제해 비교.

    표준화 계수라 "그 변수만 1σ 늘었을 때 y 가 몇 σ 움직이나" 로 크기를 직접 견줄 수 있다.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    keep = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
    X, names = X[:, keep], [names[i] for i in keep]
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(ys)), Xs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ beta
    r2 = 1 - (resid ** 2).sum() / (ys ** 2).sum()
    return list(zip(names, beta[1:])), r2


def quartiles(rows, key):
    """동점(빈도 1 이 대량)이 많아 percentile 컷이 뭉개지므로 **순위 기준**으로 4등분."""
    srt = sorted(rows, key=key)
    q = max(1, len(srt) // 4)
    labs = ["최희귀 25%", "하위 25~50%", "상위 50~75%", "최빈 25%"]
    return [(labs[i], srt[i * q:(i + 1) * q if i < 3 else len(srt)]) for i in range(4)]


def pearson(x, y):
    return float(pearsonr(x, y).statistic)


def spearman(x, y):
    return float(spearmanr(x, y).statistic)


# ─────────────────────────────────────────────────────────────────────────────
def h1(vae_json, model="korean"):
    d = json.loads(Path(vae_json).read_text())
    rows = []
    for k, v in d.get("per_syllable", {}).items():
        lbl, ch = k.split("|", 1)
        if lbl != model or not (0xAC00 <= ord(ch) <= 0xD7A3):
            continue
        miss, bot, ink, n = v
        rows.append(dict(ch=ch, miss=miss, bot=bot, ink=ink, n=n,
                         grp=jong_group(decompose(ch)[2]), st=strokes(ch), nj=n_jamo(ch)))
    if not rows:
        print("[H1] per_syllable 덤프가 없다 — jong_stroke_loss.py 를 다시 돌릴 것"); return
    print(f"=== H1: 획 수가 설명변수인가 (음절 {len(rows)}종, VAE={model}) ===\n")

    st = [r["st"] for r in rows]; ms = [r["miss"] for r in rows]
    ink = [r["ink"] for r in rows]; nj = [r["nj"] for r in rows]
    print("전체 상관 (Pearson / Spearman):")
    print(f"  획 수  vs missing : {pearson(st, ms):+.3f} / {spearman(st, ms):+.3f}")
    print(f"  자모수 vs missing : {pearson(nj, ms):+.3f} / {spearman(nj, ms):+.3f}")
    print(f"  잉크px vs missing : {pearson(ink, ms):+.3f} / {spearman(ink, ms):+.3f}")
    print(f"  획 수  vs 잉크px  : {pearson(st, ink):+.3f}\n")

    print("획 수 구간별 missing (구간 안에서 종성 갈래 비교 = 획 수 통제):")
    print(f"{'획수구간':10s} {'없음':>16s} {'홑받침':>16s} {'쌍받침':>16s} {'겹받침':>16s}")
    bins = [(0, 6), (7, 8), (9, 10), (11, 12), (13, 40)]
    ctrl = defaultdict(list)
    for lo, hi in bins:
        cells = []
        for g in ("없음", "홑받침", "쌍받침(ㄲㅆ)", "겹받침(ㄳㄵ…)"):
            sub = [r["miss"] for r in rows if lo <= r["st"] <= hi and r["grp"] == g]
            cells.append(f"{np.mean(sub):.3f}(n={len(sub)})" if sub else "-")
            if sub:
                ctrl[(lo, hi, g)] = sub
        print(f"{f'{lo}~{hi}':10s} " + " ".join(f"{c:>16s}" for c in cells))

    # 획 수를 맞춘 뒤의 겹받침 - 홑받침 차이(구간별 가중평균)
    diffs, wts = [], []
    for lo, hi in bins:
        a = ctrl.get((lo, hi, "겹받침(ㄳㄵ…)")); b = ctrl.get((lo, hi, "홑받침"))
        if a and b and len(a) >= 5 and len(b) >= 5:
            diffs.append(np.mean(a) - np.mean(b)); wts.append(min(len(a), len(b)))
    raw_a = [r["miss"] for r in rows if r["grp"] == "겹받침(ㄳㄵ…)"]
    raw_b = [r["miss"] for r in rows if r["grp"] == "홑받침"]
    print(f"\n  통제 전 겹−홑 = {np.mean(raw_a) - np.mean(raw_b):+.4f}")
    if diffs:
        print(f"  획수 통제 후 겹−홑 = {np.average(diffs, weights=wts):+.4f} "
              f"({len(diffs)}개 구간 가중평균)")
        print("  → 통제 후 차이가 0 근처면 원인은 '겹받침'이 아니라 '획 수/잉크 밀도'다.")

    print("\n획 수 구간별 '받침자리(아래40%)' missing:")
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r["st"] <= hi]
        if sub:
            print(f"  {lo:2d}~{hi:2d}: bot={np.mean([r['bot'] for r in sub]):.3f} "
                  f"miss={np.mean([r['miss'] for r in sub]):.3f} (n={len(sub)})")


# ─────────────────────────────────────────────────────────────────────────────
def h2(vae_json, gen_json=None, model="korean"):
    cnt = corpus_counts()
    total = sum(cnt.values())
    exp = train_exposure()
    tot_e = sum(exp.values())
    print(f"\n\n=== H2: 학습 노출량이 설명변수인가 ===")
    print(f"코퍼스 한글 음절 토큰 {total:,} / 고유 음절 {len(cnt):,}")
    print("⚠️ 샘플러는 코퍼스 빈도를 안 쓴다(고유 단어 균등 + rand 균등) → '실노출' 열이 진짜 학습량\n")
    print(f"{'종성유형':16s} {'코퍼스토큰%':>11s} {'실노출%':>9s} {'음절당 노출배율(홑=1)':>22s} {'음절수':>7s}")
    _per = {}
    for g in ("없음", "홑받침", "쌍받침(ㄲㅆ)", "겹받침(ㄳㄵ…)"):
        chs = [c for c in exp if jong_group(decompose(c)[2]) == g]
        e = sum(exp[c] for c in chs); cc = sum(cnt.get(c, 0) for c in chs)
        _per[g] = (cc / total * 100, e / tot_e * 100, e / max(1, len(chs)), len(chs))
    base = _per["홑받침"][2]
    for g, (a, b, per, n) in _per.items():
        print(f"{g:16s} {a:10.2f}% {b:8.2f}% {per/base:21.2f} {n:7d}")
    print()

    by_g = defaultdict(int); uniq = defaultdict(set)
    for ch, c in cnt.items():
        g = jong_group(decompose(ch)[2])
        by_g[g] += c; uniq[g].add(ch)
    print(f"{'종성유형':16s} {'토큰비율':>9s} {'고유음절':>8s} {'평균빈도':>10s}")
    for g in ("없음", "홑받침", "쌍받침(ㄲㅆ)", "겹받침(ㄳㄵ…)"):
        print(f"{g:16s} {by_g[g]/total*100:8.2f}% {len(uniq[g]):8d} "
              f"{by_g[g]/max(1,len(uniq[g])):10.1f}")

    # 종성 글자별 빈도 (겹받침이 실제로 얼마나 희귀한지)
    jc = Counter()
    for ch, c in cnt.items():
        j = decompose(ch)[2]
        if j:
            jc[j] += c
    print("\n종성 글자별 코퍼스 빈도 (희귀 순 하위 15):")
    for j, c in sorted(jc.items(), key=lambda kv: kv[1])[:15]:
        print(f"  {j}  {c:8,d} ({c/total*100:6.3f}%)  {jong_group(j)}")

    # H2b: 빈도 vs VAE missing (VAE 는 학습에 코퍼스 텍스트를 쓰므로 빈도 효과가 있을 수 있다)
    d = json.loads(Path(vae_json).read_text())
    rows = []
    for k, v in d.get("per_syllable", {}).items():
        lbl, ch = k.split("|", 1)
        if lbl == model and 0xAC00 <= ord(ch) <= 0xD7A3 and cnt.get(ch):
            rows.append((ch, v[0], cnt[ch], jong_group(decompose(ch)[2]), strokes(ch)))
    if rows:
        f = [np.log10(r[2]) for r in rows]; m = [r[1] for r in rows]
        print(f"\nVAE missing vs log10(코퍼스 빈도): Pearson {pearson(f, m):+.3f} / "
              f"Spearman {spearman(f, m):+.3f}  (n={len(rows)} 음절)")
        print("실노출 4분위별 missing (획 수도 같이 보여 교란 확인):")
        for lab, sub in quartiles(rows, key=lambda r: exp.get(r[0], 0.0)):
            print(f"  {lab:14s} missing={np.mean([r[1] for r in sub]):.3f} "
                  f"획수={np.mean([r[4] for r in sub]):4.1f} (n={len(sub)})")
        coef, r2 = ols_std([r[1] for r in rows],
                           [[r[4], np.log10(exp.get(r[0], 1e-6)), float(r[3] == "겹받침(ㄳㄵ…)")]
                            for r in rows],
                           ["획수", "log실노출", "겹받침"])
        print("\n표준화 다중회귀  VAE missing ~ 획수 + log실노출 + 겹받침 더미:")
        for nm, b in coef:
            print(f"  {nm:8s} β={b:+.3f}")
        print(f"  R²={r2:.3f}  → β 절대값이 큰 쪽이 (다른 걸 통제했을 때) 진짜 설명변수")

    if gen_json and Path(gen_json).exists():
        g = json.loads(Path(gen_json).read_text()).get("per_syllable", {})
        rows = [(ch, v[0], cnt.get(ch, 0), strokes(ch), jong_group(decompose(ch)[2]))
                for ch, v in g.items() if cnt.get(ch)]
        if rows:
            f = [np.log10(exp.get(r[0], 1e-6)) for r in rows]; a = [r[1] for r in rows]
            print(f"\n생성 음절정확도 vs log10(실노출): Pearson {pearson(f, a):+.3f} / "
                  f"Spearman {spearman(f, a):+.3f}  (n={len(rows)} 음절)")
            coef, r2 = ols_std(a, [[r[3], np.log10(exp.get(r[0], 1e-6)),
                                    float(r[4] == "겹받침(ㄳㄵ…)")] for r in rows],
                               ["획수", "log실노출", "겹받침"])
            print("표준화 다중회귀  생성 음절정확도 ~ 획수 + log실노출 + 겹받침 더미:")
            for nm, b in coef:
                print(f"  {nm:8s} β={b:+.3f}")
            print(f"  R²={r2:.3f}  (정확도가 종속 → 음수 β = 나쁘게 만드는 요인)")
            print("실노출 구간별 생성 정확도:")
            ex = sorted(exp.get(r[0], 0.0) for r in rows)
            qs = [ex[len(ex)//3], ex[2*len(ex)//3]]
            for lo, hi, lab in ((0, qs[0], "하위33%"), (qs[0], qs[1], "중간"), (qs[1], 1e18, "상위33%")):
                sub = [r for r in rows if lo <= exp.get(r[0], 0.0) < hi]
                if sub:
                    print(f"  {lab:8s} 정확도={np.mean([r[1] for r in sub]):.3f} "
                          f"획수={np.mean([r[3] for r in sub]):4.1f} (n={len(sub)})")
            print("\n실노출을 맞춘 뒤 겹받침 vs 홑받침 생성 정확도:")
            for lo, hi, lab in ((0, qs[0], "하위33%"), (qs[0], qs[1], "중간"), (qs[1], 1e18, "상위33%")):
                cl = [r[1] for r in rows if lo <= exp.get(r[0], 0.0) < hi and r[4] == "겹받침(ㄳㄵ…)"]
                si = [r[1] for r in rows if lo <= exp.get(r[0], 0.0) < hi and r[4] == "홑받침"]
                if cl and si:
                    print(f"  {lab:8s} 겹={np.mean(cl):.3f}(n={len(cl)}) "
                          f"홑={np.mean(si):.3f}(n={len(si)})  차이={np.mean(cl)-np.mean(si):+.3f}")
    else:
        print("\n[H2b] --gen-json 미제공 → 생성 정확도 vs 빈도 상관은 생략")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-json", required=True)
    ap.add_argument("--gen-json", default=None)
    ap.add_argument("--model", default="korean")
    args = ap.parse_args()
    h1(args.vae_json, args.model)
    h2(args.vae_json, args.gen_json, args.model)


if __name__ == "__main__":
    main()
