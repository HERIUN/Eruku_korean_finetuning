"""폰트 풀을 학습/미노출로 결정적 분할 — v3/train → v4/{train,heldout}.

지금까지 미노출 세트가 인쇄체 16종과 실제필적 4종뿐이라, 학습 분포(손글씨 12,951종)와
같은 종류의 미노출이 하나도 없었다. 온글잎/눈누에서 일부를 떼어 그걸 만든다.
심볼릭 링크만 만든다(복사하면 15GB 중복).
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def source_of(p: Path) -> str:
    t = str(p.resolve())
    for k in ("ownglyph", "noonnu"):
        if f"/fonts_korean_new/{k}/" in t:
            return k
    return "v2train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/fonts_korean_v3/train")
    ap.add_argument("--out", default="assets/fonts_korean_v4")
    ap.add_argument("--n-ownglyph", type=int, default=500)
    ap.add_argument("--n-noonnu", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()

    src = HERE / a.src
    fonts = sorted(p for p in src.glob("*") if p.suffix.lower() in (".ttf", ".otf"))
    by = {}
    for p in fonts:
        by.setdefault(source_of(p), []).append(p)
    print("소스별:", {k: len(v) for k, v in sorted(by.items())})

    rng = random.Random(a.seed)
    held = []
    for k, n in (("ownglyph", a.n_ownglyph), ("noonnu", a.n_noonnu)):
        pool = sorted(by.get(k, []))
        if n > len(pool):
            raise SystemExit(f"{k}: {n} > 보유 {len(pool)}")
        held += rng.sample(pool, n)
    held_set = {p.name for p in held}
    train = [p for p in fonts if p.name not in held_set]

    for sub, items in (("train", train), ("heldout", held)):
        d = HERE / a.out / sub
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*"):
            if old.is_symlink():
                old.unlink()
        for p in items:
            (d / p.name).symlink_to(p.resolve())
        print(f"{d.relative_to(HERE)}: {len(items)}종")

    # 재현을 위해 미노출 목록을 파일로 남긴다 — 이 목록이 평가셋의 정체다.
    man = HERE / a.out / "heldout_fonts.txt"
    man.write_text("\n".join(sorted(p.name for p in held)) + "\n", encoding="utf-8")
    print(f"목록 저장 {man.relative_to(HERE)} (seed={a.seed})")


if __name__ == "__main__":
    main()
