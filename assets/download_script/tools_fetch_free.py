"""noonnu(눈누) / ownglyph(온글잎) 무료 한글 폰트 수집기. `--src noonnu|ownglyph`

noonnu:   sitemap.xml → font_page 전체 → 페이지에 박힌 jsdelivr webfont(woff2) 를 받아 TTF/OTF 로 변환.
          (다운로드 버튼은 제조사 사이트로 나가서 자동화 불가 → webfont 가 유일한 자동 경로)
          페이지의 라이선스 요약표·제조사 다운로드 URL 도 같이 카탈로그에 기록한다.
ownglyph: 공개 Firestore REST 로 폰트 목록 조회 → `fontUrl`(S3 TTF 직링크) 다운로드.
          ⚠ 이 컬렉션에는 폰트 주인의 이메일·전화번호·생년월일이 들어 있다. `mask.fieldPaths` 로
          폰트 관련 필드만 요청해 개인정보는 애초에 받지도 저장하지도 않는다.

공통: 기존 보유 폰트(fonts_korean_v2/train·test, custom_hw_fonts, fonts_english)와
      ① 이름 정규화 ② 글리프 아웃라인 해시 2단으로 중복 제거하고, KS X 1001 표본 커버리지로 거른다.
      woff2 → sfnt 재컴파일이 폰트당 ~3초 CPU 라 워커는 스레드가 아니라 **프로세스 풀**이다.

사용:
  .venv/bin/python assets/download_script/tools_fetch_free.py --src noonnu
  .venv/bin/python assets/download_script/tools_fetch_free.py --src ownglyph --n 1000
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time, urllib.parse
from concurrent.futures import ProcessPoolExecutor
from html import unescape
from io import BytesIO
from pathlib import Path

from fontTools.ttLib import TTFont, woff2
from fontTools.pens.recordingPen import DecomposingRecordingPen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools_fetch_gf import HERE, KS_SAMPLE, cmap_of, get, norm  # noqa: E402,F401

SITEMAP = "https://noonnu.cc/sitemap.xml"
FS_BASE = ("https://firestore.googleapis.com/v1/projects/v6x-ownglyph/databases/(default)"
           "/documents/fonts/fontMenu/ownglyphFontV2")
FS_KEY = "AIzaSyD8olPPWNKBr87Ewi-LGFWr-nqkzzzbV-Q"
# 폰트 필드만. 나머지(notiEmail/notiPhoneNumber/copyrightOwnerBirthDate …)는 개인정보라 요청하지 않는다.
FS_FIELDS = ["fontUID", "fontNameKor", "fontNameEng", "fontUrl", "fontStyle",
             "fontIndex", "allowOwnglyphDeploy", "completed", "canceled"]

FP_CHARS = "가나다라마바사아자차카타파하각간갈감값곬넋A0"   # 지문용 대표 글자
FP_MIN = 8                                                # 이만큼도 못 그리면 지문 포기
EXISTING_DIRS = ["assets/fonts_korean_v2/train", "assets/fonts_korean_v2/test",
                 "assets/custom_hw_fonts", "assets/fonts_english"]
FONT_URL_RE = re.compile(r"url\(\s*['\"]?([^)'\"\s]+\.(?:woff2|woff|ttf|otf))", re.I)
CSS_URL_RE = re.compile(r"url\(\s*['\"]?([^)'\"\s]+\.css)", re.I)
HEAVY = ("bold", "black", "heavy", "extra", "thin", "light", "italic")
LIGHT_SUF = ("regular", "rg", "medium", "md", "m", "r")   # GodoM(Medium) 이 GodoB 보다 본문체

_KNOWN = (set(), set())     # 워커 프로세스가 들고 있는 (이름, 지문) 스냅샷 — 중복이면 변환 자체를 건너뛴다


# ── 폰트 검사·변환 (CPU 무거운 부분: 프로세스 워커에서 실행) ──────────────────
def glyph_fp(font: TTFont):
    """대표 글자 아웃라인을 upm 정규화해 해시. 파일명이 달라도 같은 폰트면 같은 값."""
    try:
        cmap, gs = font.getBestCmap(), font.getGlyphSet()
        upm = font["head"].unitsPerEm or 1000
    except Exception:
        return None
    h, n = hashlib.sha1(), 0
    for ch in FP_CHARS:
        gn = cmap.get(ord(ch))
        if not gn or gn not in gs:
            continue
        pen = DecomposingRecordingPen(gs)
        try:
            gs[gn].draw(pen)
        except Exception:
            continue
        parts = []
        for op, args in pen.value:
            parts.append(op)
            for a in args:
                parts.append(f"{a[0]/upm:.4f},{a[1]/upm:.4f}" if isinstance(a, tuple) else str(a))
        h.update(("|".join(parts) + "##").encode())
        n += 1
    return h.hexdigest() if n >= FP_MIN else None


def inspect_convert(data: bytes, stem: str, min_cov: float):
    """bytes → 검사(커버리지·중복) → 필요 시 sfnt 로 재컴파일.

    반환 dict: status / cov / fp / space / ext / bytes.
    무거운 재컴파일 전에 커버리지·중복을 먼저 걸러 헛일을 줄인다.
    ownglyph 처럼 이미 평문 sfnt 면 원본 bytes 를 그대로 쓴다(재컴파일 생략).
    """
    try:
        f = TTFont(BytesIO(data), fontNumber=0)
    except Exception as e:
        return {"status": f"BADFONT({type(e).__name__})", "cov": 0.0}
    cm = f.getBestCmap() or {}
    sample = list(dict.fromkeys(ord(c) for c in KS_SAMPLE))
    cov = sum(1 for cp in sample if cp in cm) / len(sample)
    if cov < min_cov:
        return {"status": "LOWCOV", "cov": cov}
    fp = glyph_fp(f)
    if norm(stem) in _KNOWN[0] or (fp and fp in _KNOWN[1]):
        return {"status": "SKIP-DUP", "cov": cov, "fp": fp}
    ext = ".ttf" if "glyf" in f else ".otf"
    if f.flavor:                              # woff/woff2 → 평문 sfnt (여기가 폰트당 ~3초)
        flavor, raw = f.flavor, data
        f.flavor = None
        buf = BytesIO()
        try:
            f.save(buf)
            data = buf.getvalue()
        except Exception as e:
            if flavor != "woff2":             # fontTools 재컴파일이 깨지는 폰트가 소수 있다
                return {"status": f"SAVEFAIL({type(e).__name__})", "cov": cov}
            buf = BytesIO()                   # → 저수준 woff2 복원으로 한 번 더 시도
            try:
                woff2.decompress(BytesIO(raw), buf)
                data = buf.getvalue()
            except Exception as e2:
                return {"status": f"SAVEFAIL({type(e2).__name__})", "cov": cov}
    return {"status": "OK", "cov": cov, "fp": fp, "space": 0x20 in cm,
            "ext": ext, "bytes": data}


# ── 보유 폰트 지문 인덱스 ────────────────────────────────────────────────────
def build_existing_index(extra_dir=None):
    """보유 폰트 → (정규화 이름 집합, 글리프 지문 집합).

    extra_dir(=수집 staging) 도 함께 훑어 재실행 시 이어받기와 소스 간 교차 중복 제거가 된다.
    """
    names, fps = set(), set()
    paths = [q for d in EXISTING_DIRS for q in Path(HERE / d).glob("*.[to]tf")]
    if extra_dir:
        paths += list(Path(extra_dir).rglob("*.[to]tf"))
    for p in sorted(paths):
        names.add(norm(p.stem))
        try:
            fp = glyph_fp(TTFont(str(p), fontNumber=0))
        except Exception:
            fp = None
        if fp:
            fps.add(fp)
    return names, fps


def _init_worker(names, fps):
    _KNOWN[0].update(names)
    _KNOWN[1].update(fps)


# ── noonnu ──────────────────────────────────────────────────────────────────
def _abs(u, base=None):
    """`//cdn…` · 상대경로 → 절대 https URL."""
    if u.startswith("//"):
        return "https:" + u
    if u.startswith(("http://", "https://")):
        return u
    return urllib.parse.urljoin(base, u) if base else None


def _face_blocks(css, base=None):
    """CSS 텍스트의 @font-face → [(family, 폰트파일 절대URL)]"""
    out = []
    for m in re.finditer(r"@font-face\s*\{(.*?)\}", css, re.S):
        block = m.group(1)
        fam = re.search(r"font-family:\s*['\"]?([^;'\"]+)", block)
        fam = fam.group(1).strip() if fam else ""
        for um in FONT_URL_RE.finditer(block):
            u = _abs(um.group(1), base)
            if u:
                out.append((fam, u))
    return out


def weight_rank(url):
    """Regular 계열 우선 정렬 키(기존 tools_fetch_gf.fetch_korean 의 score() 와 같은 취지)."""
    n = url.rsplit("/", 1)[-1].lower()
    stem = n.rsplit(".", 1)[0]
    tier = 0 if "regular" in stem else (1 if not any(h in stem for h in HEAVY) else 2)
    suf = 0 if any(stem.endswith(x) for x in LIGHT_SUF) else 1
    return (tier, suf, len(n), n)


def parse_noonnu_page(html):
    """폰트 페이지 → {name_kor, webfonts:[(family,url)], license:{cat:allow}, vendor_url}

    페이지 본문의 @font-face 는 그 폰트 전용이다(사이트 UI 폰트는 외부 CSS 로 따로 로드).
    일부 구형 폰트는 @font-face 대신 제조사 CSS 를 import 하므로 그 경우 CSS 를 한 단계 더 따라간다.
    """
    title = re.search(r"<title>(.*?)</title>", html, re.S)
    name_kor = unescape(title.group(1)).split("|")[0].strip() if title else ""
    txt = unescape(html)

    webfonts, seen = [], set()
    for fam, url in _face_blocks(txt):
        if url not in seen:
            seen.add(url)
            webfonts.append((fam, url))
    if not webfonts:
        for cu in dict.fromkeys(CSS_URL_RE.findall(txt)):
            cu = _abs(cu)
            css = get(cu) if cu else None
            if not css:
                continue
            for fam, url in _face_blocks(css, base=cu):
                if url not in seen:
                    seen.add(url)
                    webfonts.append((fam, url))

    lic = {}
    seg = html[html.find("라이선스 요약표"):]
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", seg, re.S)[:12]:
        cells = [unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c))).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        if len(cells) >= 3 and cells[0] not in ("카테고리", ""):
            lic[cells[0]] = cells[2]

    vend = re.search(r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*noon-yellow-button', html) or \
        re.search(r'<a[^>]+class="[^"]*noon-yellow-button[^"]*"[^>]*href="([^"]+)"', html)
    return {"name_kor": name_kor, "webfonts": webfonts, "license": lic,
            "vendor_url": vend.group(1) if vend else None}


def noonnu_item(task):
    """워커: 페이지 1건 → (catalog, [폰트 결과]). 예외는 삼켜서 전체 실행을 지키기."""
    page, out_dir, all_weights, sleep, min_cov = task
    try:
        html = get(page, retries=5)      # noonnu 는 동시요청에 민감 — 재시도를 넉넉히
        if not html:
            return None, [{"page": page, "status": "PAGEFAIL"}]
        info = parse_noonnu_page(html)
        cat = {"page": page, "name_kor": info["name_kor"],
               "vendor_download_url": info["vendor_url"], "license": info["license"],
               "webfonts": [{"family": f, "url": u} for f, u in info["webfonts"]]}
        wfs = info["webfonts"]
        if not wfs:
            return cat, [{"page": page, "name": info["name_kor"], "status": "NOWEBFONT"}]
        if not all_weights:
            wfs = [min(wfs, key=lambda x: weight_rank(x[1]))]

        rows = []
        for fam, url in wfs:
            stem = re.sub(r"[^0-9A-Za-z_\-]", "", fam)
            if len(stem) < 3:                 # 한글 family("행복고흥M" → "M") → 페이지 id 로
                stem = f"noonnu{page.rsplit('/', 1)[1]}"
            if any(os.path.exists(os.path.join(out_dir, stem + e)) for e in (".ttf", ".otf")):
                rows.append({"page": page, "name": info["name_kor"], "file": stem,
                             "status": "SKIP-EXIST"})
                continue
            data = get(urllib.parse.quote(url, safe=":/?&=@%"), raw=True)
            r = {"status": "DLFAIL", "cov": 0.0} if not data \
                else inspect_convert(data, stem, min_cov)
            r.update({"page": page, "name": info["name_kor"], "file": stem, "url": url})
            rows.append(r)
        time.sleep(sleep)
        return cat, rows
    except Exception as e:
        return None, [{"page": page, "status": f"ERROR({type(e).__name__})"}]


def fetch_noonnu(args, idx):
    out = Path(args.out) / "noonnu"
    out.mkdir(parents=True, exist_ok=True)
    RETRYABLE = {"DLFAIL", "SAVEFAIL", "BADFONT", "PAGEFAIL", "ERROR"}
    keep_rows, old_cat = [], {}
    if args.retry:                      # 실패한 페이지만 재시도하고 나머지는 이전 결과를 승계
        old = json.loads(Path(args.retry).read_text(encoding="utf-8"))
        redo = {r["page"] for r in old
                if r.get("page") and r["status"].split("(")[0] in RETRYABLE}
        keep_rows = [r for r in old if r.get("page") not in redo]
        pages = sorted(redo, key=lambda u: int(u.rsplit("/", 1)[1]))
        cp = Path(args.out) / "_catalog_noonnu.json"
        if cp.exists():
            old_cat = {c["page"]: c for c in json.loads(cp.read_text(encoding="utf-8"))}
        print(f"재시도 대상 {len(pages)} 페이지 (승계 {len(keep_rows)}행)", flush=True)
    else:
        sm = get(SITEMAP) or ""
        pages = sorted(set(re.findall(r"<loc>(https://noonnu\.cc/font_page/\d+)</loc>", sm)),
                       key=lambda u: int(u.rsplit("/", 1)[1]))
        if args.limit:
            pages = pages[:args.limit]
        print(f"noonnu font_page: {len(pages)}", flush=True)

    tasks = [(p, str(out), args.all_weights, args.sleep, args.min_cov) for p in pages]
    catalog, report = [], list(keep_rows)
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker, initargs=idx) as ex:
        for i, (cat, rows) in enumerate(ex.map(noonnu_item, tasks), 1):
            if cat:
                catalog.append(cat)
            for r in rows:
                report.append(commit(r, out, idx, src="noonnu"))
                print(f"  [{i}/{len(pages)}] [{report[-1]['status']}] "
                      f"{r.get('name', '')} / {r.get('file', '')} cov={r.get('cov', 0):.2f}",
                      flush=True)

    if old_cat:                        # 재시도분으로 갱신하되 나머지 카탈로그는 보존
        old_cat.update({c["page"]: c for c in catalog})
        catalog = [old_cat[k] for k in sorted(old_cat, key=lambda u: int(u.rsplit("/", 1)[1]))]
    (Path(args.out) / "_catalog_noonnu.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


# ── ownglyph ────────────────────────────────────────────────────────────────
def _val(field):
    """Firestore Value → python scalar."""
    (k, v), = field.items()
    return int(v) if k == "integerValue" else v


def ownglyph_catalog():
    mask = "".join(f"&mask.fieldPaths={f}" for f in FS_FIELDS)
    token, rows, total = None, [], 0
    while True:
        url = (f"{FS_BASE}?key={FS_KEY}&pageSize=300{mask}"
               + (f"&pageToken={urllib.parse.quote(token)}" if token else ""))
        page = get(url)
        if not page:
            print("  [WARN] Firestore 페이지 조회 실패 — 여기까지로 중단")
            break
        d = json.loads(page)
        docs = d.get("documents", [])
        total += len(docs)
        for doc in docs:
            f = {k: _val(v) for k, v in doc.get("fields", {}).items()}
            if f.get("allowOwnglyphDeploy") and f.get("completed") and not f.get("canceled") \
                    and f.get("fontUrl"):
                rows.append({k: f.get(k) for k in FS_FIELDS})
        token = d.get("nextPageToken")
        if not token:
            break
    print(f"ownglyph docs {total} → 배포가능 {len(rows)}", flush=True)
    return rows


def ownglyph_item(task):
    """워커: 폰트 1건 다운로드 + 검사. ownglyph 는 이미 TTF 라 재컴파일이 없다."""
    r, out_dir, sleep, min_cov = task
    try:
        base = re.sub(r"[^0-9A-Za-z_\-]", "", (r.get("fontNameEng") or "")) or r["fontUID"][:8]
        stem = base
        if any(os.path.exists(os.path.join(out_dir, base + e)) for e in (".ttf", ".otf")):
            stem = f"{base}_{r['fontUID'][:8]}"
            if any(os.path.exists(os.path.join(out_dir, stem + e)) for e in (".ttf", ".otf")):
                return {"fontUID": r["fontUID"], "name": r.get("fontNameKor"),
                        "file": stem, "status": "SKIP-EXIST"}
        data = get(urllib.parse.quote(r["fontUrl"], safe=":/"), raw=True)
        res = {"status": "DLFAIL", "cov": 0.0} if not data \
            else inspect_convert(data, stem, min_cov)
        res.update({"fontUID": r["fontUID"], "name": r.get("fontNameKor"), "file": stem})
        time.sleep(sleep)
        return res
    except Exception as e:
        return {"fontUID": r.get("fontUID"), "name": r.get("fontNameKor"),
                "status": f"ERROR({type(e).__name__})"}


def fetch_ownglyph(args, idx):
    out = Path(args.out) / "ownglyph"
    out.mkdir(parents=True, exist_ok=True)
    rows = ownglyph_catalog()
    (Path(args.out) / "_catalog_ownglyph.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.catalog_only:
        return []

    rows.sort(key=lambda r: (r.get("fontIndex") or 0, r["fontUID"]))
    n = min(args.n, len(rows))
    picked = rows[::max(1, len(rows) // n)][:n]
    if args.limit:
        picked = picked[:args.limit]
    print(f"균등 샘플 {len(picked)} / {len(rows)}", flush=True)

    tasks = [(r, str(out), args.sleep, args.min_cov) for r in picked]
    report = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker, initargs=idx) as ex:
        for i, res in enumerate(ex.map(ownglyph_item, tasks), 1):
            report.append(commit(res, out, idx, src="ownglyph"))
            print(f"  [{i}/{len(picked)}] [{report[-1]['status']}] "
                  f"{res.get('name', '')} / {res.get('file', '')} cov={res.get('cov', 0):.2f}",
                  flush=True)
    return report


# ── 부모 쪽 확정(중복 최종 판정 + 파일 쓰기) ─────────────────────────────────
def commit(res, out: Path, idx, src: str):
    """워커 결과를 받아 중복 최종 판정 후 저장. 리포트 한 줄을 돌려준다."""
    names, fps = idx
    st, stem, fp = res.get("status"), res.get("file"), res.get("fp")
    if st == "OK":
        if norm(stem) in names or (fp and fp in fps):    # 이번 실행 중 새로 들어온 것과의 중복
            st = "SKIP-DUP"
        else:
            (out / f"{stem}{res['ext']}").write_bytes(res["bytes"])
            names.add(norm(stem))
            if fp:
                fps.add(fp)
    row = {"src": src, "file": stem, "name": res.get("name"),
           "coverage": round(res.get("cov", 0.0), 3), "space_glyph": res.get("space"),
           "status": st}
    for k in ("page", "url", "fontUID"):
        if k in res:
            row[k] = res[k]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["noonnu", "ownglyph"], required=True)
    ap.add_argument("--out", default=str(HERE / "assets/fonts_korean_new"))
    ap.add_argument("--min-cov", type=float, default=0.95, help="KS 표본 커버리지 최소(미만 제외)")
    ap.add_argument("--n", type=int, default=1000, help="[ownglyph] 균등 샘플 개수")
    ap.add_argument("--limit", type=int, default=0, help="디버그용 상한(0=제한없음)")
    ap.add_argument("--sleep", type=float, default=0.3, help="워커별 요청 간격(초)")
    ap.add_argument("--workers", type=int, default=6, help="동시 워커(프로세스) 수")
    ap.add_argument("--all-weights", action="store_true", help="[noonnu] 굵기별 전부 받기")
    ap.add_argument("--catalog-only", action="store_true", help="[ownglyph] 목록만 저장하고 종료")
    ap.add_argument("--retry", default=None,
                    help="[noonnu] 이전 리포트 json 의 실패 페이지만 재시도(카탈로그·리포트는 병합)")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    print("보유 폰트 지문 인덱스 작성 중…", flush=True)
    idx = build_existing_index(args.out)
    print(f"  이름 {len(idx[0])}종 / 지문 {len(idx[1])}종", flush=True)

    report = (fetch_noonnu if args.src == "noonnu" else fetch_ownglyph)(args, idx)

    rp = Path(args.out) / f"_fetch_report_{args.src}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    tally = {}
    for r in report:
        key = r["status"].split("(")[0]
        tally[key] = tally.get(key, 0) + 1
    nospace = sum(1 for r in report if r["status"] == "OK" and r.get("space_glyph") is False)
    print(f"\n집계: {tally}")
    if nospace:
        print(f"주의: 공백(U+0020) 글리프 없는 폰트 {nospace}종 — 리포트의 space_glyph=false 로 확인")
    print(f"리포트: {rp}")


if __name__ == "__main__":
    main()
