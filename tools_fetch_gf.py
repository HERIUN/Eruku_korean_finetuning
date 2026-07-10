"""google/fonts 폰트 수집기 (한글/영어 공용). `--lang ko|en`

ko: korean subset family → github API 로 ttf 탐색 → KS X 1001(2350자) 커버리지 필터.
    기존 보유(fonts_korean_v2/train + fonts_korean_unseen)와 중복 제거. staging 디렉토리에 저장.
en: korean 미포함 + latin 포함 family 중 Handwriting/Display 우선(손글씨 도메인 직결)
    → raw URL 추정 다운로드(API rate-limit 회피) → 라틴 소문자 커버리지 확인.

사용:
  .venv/bin/python tools_fetch_gf.py --lang ko --out assets/fonts_korean_gf_new
  .venv/bin/python tools_fetch_gf.py --lang en --out assets/fonts_english --n-hand 150 --n-disp 50
"""
from __future__ import annotations
import argparse, json, re, time, urllib.request
from pathlib import Path
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (font-fetch)"}
RAW = "https://raw.githubusercontent.com/google/fonts/main"
META_URL = "https://fonts.google.com/metadata/fonts"

# KS X 1001 대표 음절 표본(빠른 한글 커버리지 측정용)
KS_SAMPLE = ("가나다라마바사아자차카타파하각간갈감갑강개객거건걸검겁것게겨견결경계고곡곤골공과관광교"
             "구국군굴궁권귀규그근글금급기긴길김까꽃끝나는님다도되라로를리만말명물및바방배보부분비사"
             "상새서선설성세소수시안아어업에여연영오와요용우원월위유은을의이인일자작전점정제조주중지직"
             "진짜차참천체초총최카타파평포하학한해형호화황회")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get(url, raw=False, retries=3):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read() if raw else r.read().decode("utf-8")
        except Exception:
            if i == retries - 1:
                return None if raw else None
            time.sleep(2)


def cmap_of(font_path):
    try:
        return TTFont(str(font_path)).getBestCmap()
    except Exception:
        return None


def load_meta(meta_cache=None):
    if meta_cache and Path(meta_cache).exists():
        return json.loads(Path(meta_cache).read_text())
    return json.loads(get(META_URL))


# ── 한글 ────────────────────────────────────────────────────────────────────
def fetch_korean(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    existing = {norm(p.stem) for p in (HERE / "assets/fonts_korean_v2/train").glob("*.ttf")}
    existing |= {norm(p.stem) for p in (HERE / "assets/fonts_korean_unseen").glob("*.ttf")}

    meta = load_meta(args.meta)
    kor = [f for f in meta["familyMetadataList"] if "korean" in (f.get("subsets") or [])]
    new = [f for f in kor if norm(f["family"]) not in existing]
    print(f"korean subset {len(kor)} | 기존중복 {len(kor)-len(new)} | 신규 {len(new)}")

    sample = list(dict.fromkeys(ord(c) for c in KS_SAMPLE))
    report = []
    for f in sorted(new, key=lambda x: x["family"]):
        fam = f["family"]; d = norm(fam)
        listing = get(f"https://api.github.com/repos/google/fonts/contents/ofl/{d}")
        listing = json.loads(listing) if listing else None
        if not isinstance(listing, list):
            print(f"  [SKIP] {fam}: ofl/{d} 없음"); continue
        ttfs = [x for x in listing if x["name"].lower().endswith(".ttf")]
        if not ttfs:
            print(f"  [SKIP] {fam}: ttf 없음"); continue

        def score(n):                       # 정적 Regular 우선 → 가변([wght]) → 나머지
            n2 = n.lower()
            return 0 if ("regular" in n2 and "[" not in n2) else (1 if "[" in n2 else 2)
        pick = sorted(ttfs, key=lambda x: (score(x["name"]), len(x["name"])))[0]
        dst = out / f"{fam.replace(' ', '')}.ttf"
        data = get(pick["download_url"], raw=True)
        if not data:
            print(f"  [SKIP dl] {fam}"); continue
        dst.write_bytes(data)
        cm = cmap_of(dst) or {}
        cov = sum(1 for cp in sample if cp in cm) / len(sample)
        status = "OK" if cov >= args.min_cov else "LOWCOV"
        if cov < args.min_cov:
            dst.unlink(missing_ok=True)
        report.append((fam, fam.replace(" ", ""), f.get("category"), cov, status, pick["name"]))
        print(f"  [{status}] {fam:24s} cov={cov:.2f} <- {pick['name']}")

    json.dump([{"family": r[0], "file": r[1], "category": r[2], "coverage": r[3],
                "status": r[4], "src": r[5]} for r in report],
              open(out / "_fetch_report.json", "w"), ensure_ascii=False, indent=1)
    ok = [r for r in report if r[4] == "OK"]
    print(f"\n받은 폰트(OK): {len(ok)} / 시도 {len(report)}  -> {out}")


# ── 영어(라틴) ────────────────────────────────────────────────────────────────
def fetch_english(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(args.meta)
    cand = [f for f in meta["familyMetadataList"]
            if "korean" not in (f.get("subsets") or []) and "latin" in (f.get("subsets") or [])]
    hand = [f for f in cand if f.get("category") == "Handwriting"][:args.n_hand]
    disp = [f for f in cand if f.get("category") == "Display"][:args.n_disp]
    pick = hand + disp
    print(f"대상: handwriting {len(hand)} + display {len(disp)} = {len(pick)}")

    ok = 0
    for f in pick:
        fam = f["family"]; d = norm(fam); fn = fam.replace(" ", "")
        dst = out / f"{fn}.ttf"
        if dst.exists():
            ok += 1; continue
        data = None
        for url in (f"{RAW}/ofl/{d}/{fn}-Regular.ttf", f"{RAW}/ofl/{d}/{fn}[wght].ttf",
                    f"{RAW}/ofl/{d}/static/{fn}-Regular.ttf", f"{RAW}/ofl/{d}/{fn}[opsz,wght].ttf"):
            data = get(url, raw=True)
            if data:
                break
        if not data:
            print(f"  [miss] {fam}"); continue
        dst.write_bytes(data)
        cm = cmap_of(dst) or {}
        if not (ord("a") in cm and ord("A") in cm and ord("g") in cm):
            dst.unlink(missing_ok=True); print(f"  [nolatin] {fam}"); continue
        ok += 1
    print(f"\n받은 영어 폰트: {ok} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["ko", "en"], required=True)
    ap.add_argument("--out", default=None, help="저장 디렉토리(미지정 시 lang별 기본)")
    ap.add_argument("--meta", default=None, help="metadata json 캐시 경로(있으면 재사용, 없으면 온라인)")
    ap.add_argument("--min-cov", type=float, default=0.95, help="[ko] 한글 커버리지 최소(이하 제외)")
    ap.add_argument("--n-hand", type=int, default=150, help="[en] Handwriting 최대 수")
    ap.add_argument("--n-disp", type=int, default=50, help="[en] Display 최대 수")
    args = ap.parse_args()
    if args.out is None:
        args.out = str(HERE / "assets" / ("fonts_korean_gf_new" if args.lang == "ko" else "fonts_english"))
    (fetch_korean if args.lang == "ko" else fetch_english)(args)


if __name__ == "__main__":
    main()
