"""한글 코퍼스 확장 — 한국어 위키백과 문장에서 자연어 라인을 뽑아 `korean_lines.txt` 를 키운다.

왜: 기존 코퍼스는 고유 941줄 / 어절 5,196 / 음절 754자뿐이라, 자주 쓰는 2,350 음절 중 **68%가
자연어 문맥에 한 번도 안 나온다**(오직 rand 토큰의 무작위 나열로만 등장). docs/DATA_LIMITATIONS.md D10.

안전장치 2개:
  ① **고정 charset 밖 글자는 통째로 버린다.** 알파벳은 `assets/korean_charset.json` 으로 고정돼 있고
     HTR/VAE ckpt 이 alphabet_size=2512 에 핀 고정이다. 새 글자가 들어오면 리더가 못 읽어 CER 이 헛뜬다.
     폰트 교집합(렌더 가능 집합)과도 교차해, 렌더 단계에서 조용히 삭제되는 글자를 애초에 안 넣는다.
  ② **평가 텍스트는 이미 분리돼 있다**(`assets/corpus/eval_lines_ko.txt`). 코퍼스를 키워도 지표는 안 깨진다.

출처: 한국어 위키백과 (CC BY-SA 4.0), HF `heegyu/kowiki-sentences`. 출처 표기는 assets/corpus/README.md.

사용:
  .venv/bin/python tools/fetch_corpus.py --n-lines 50000
"""
from __future__ import annotations
import argparse, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
CORPUS = HERE / "assets" / "corpus"
HANGUL = re.compile(r"[가-힣]")

REPO_ID, FILENAME = "heegyu/kowiki-sentences", "wiki_20221001_sents.parquet"


def allowed_chars() -> set[str]:
    """고정 알파벳 ∩ 폰트 교집합(렌더 가능) + 공백. 이 밖의 글자가 있는 줄은 버린다."""
    from custom_datasets.korean.alphabet import load_charset
    from custom_datasets.korean.split import font_charset_cps, DEFAULT_EXCLUDE_FONTS, data_paths
    fonts_dir = Path(data_paths(None)["korean_fonts_dir"])
    csj = fonts_dir / "fonts_charsets.json"
    render_ok = {chr(c) for c in font_charset_cps(csj, "intersection",
                                                  exclude=DEFAULT_EXCLUDE_FONTS)[0]}
    return (set(load_charset()) & render_ok) | {" "}


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def usable(line: str, ok: set[str], lo: int, hi: int, min_hangul: float) -> bool:
    if not (lo <= len(line) <= hi) or len(line.split()) < 2:
        return False
    if not set(line) <= ok:
        return False
    return len(HANGUL.findall(line)) / len(line) >= min_hangul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lines", type=int, default=50000, help="최종 신규 라인 수(커버리지분 포함)")
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=70, help="파이프라인이 짧은 라인 가정 위에 있다")
    ap.add_argument("--min-hangul", type=float, default=0.6, help="한글 비율 하한(숫자·영문 범벅 배제)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(CORPUS / "korean_lines.txt"))
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 통계만")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    ok = allowed_chars()
    print(f"허용 글자 {len(ok)}자 (고정 알파벳 ∩ 폰트 교집합)")

    print(f"내려받는 중: {REPO_ID}/{FILENAME} …", flush=True)
    fp = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset")
    tbl = pq.read_table(fp)
    col = next((c for c in ("sentence", "text", "content") if c in tbl.column_names),
               tbl.column_names[0])          # kowiki-sentences 는 title/pi/si/sentence 스키마다
    print(f"  {tbl.num_rows:,}행, 컬럼 {tbl.column_names} → '{col}' 사용")

    seen, cand = set(), []
    for s in tbl.column(col).to_pylist():
        s = clean(s or "")
        if s in seen or not usable(s, ok, a.min_len, a.max_len, a.min_hangul):
            continue
        seen.add(s); cand.append(s)
    print(f"필터 통과 고유 라인 {len(cand):,}")

    base = [l for l in Path(a.out).read_text(encoding="utf-8").splitlines() if l.strip()]
    have = {c for l in base for c in l if HANGUL.match(c)}
    target = {c for c in ok if HANGUL.match(c)}
    print(f"기존 {len(base)}줄, 음절 {len(have)}/{len(target)}")

    # ① 커버리지 패스 — 새 음절을 가장 많이 들여오는 줄부터 (희귀 음절은 이 패스로만 들어온다)
    rng = random.Random(a.seed)
    rng.shuffle(cand)
    picked, cov = [], set(have)
    for s in cand:
        new = {c for c in s if c in target} - cov
        if new:
            picked.append(s); cov |= new
            if cov >= target:
                break
    print(f"커버리지 패스: {len(picked)}줄 → 음절 {len(cov)}/{len(target)}")

    # ② 물량 패스 — 어휘 폭. 커버리지분과 합쳐 n_lines 까지
    rest = [s for s in cand if s not in set(picked)]
    picked += rest[:max(0, a.n_lines - len(picked))]
    print(f"최종 신규 {len(picked)}줄")

    merged = base + [s for s in picked if s not in set(base)]
    syl = {c for l in merged for c in l if HANGUL.match(c)}
    eoj = {w for l in merged for w in l.split() if HANGUL.search(w)}
    print(f"\n=== 결과 ===\n라인 {len(base)} → {len(merged)}"
          f"\n음절 {len(have)} → {len(syl)} / {len(target)} ({len(syl)/len(target)*100:.1f}%)"
          f"\n어절 {len({w for l in base for w in l.split() if HANGUL.search(w)})} → {len(eoj)}")
    if a.dry_run:
        print("(dry-run — 파일 안 씀)"); return
    Path(a.out).write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
