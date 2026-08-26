"""폰트 풀 게이트: 학습에 넣어도 되는 폰트인지 판정하고, 원하면 제외 디렉토리로 옮긴다.

`font_audit.py`(기하 이상)와 `font_charset.py`(실렌더 charset)를 재사용해 한 번에 판정한다.
`charset_rule=intersection` 인 한글 파이프라인은 **폰트 한 종이 풀 전체를 끌어내리므로**
개별 폰트 품질뿐 아니라 **교집합 영향**까지 봐야 한다(실측: 눈누 1,025종 그대로 두면 한글 교집합이
2,350 → 496자로 무너지고, 그중 2종이 피해의 대부분이었다).

판정
  REJECT   폰트 로드·렌더 실패 / 한글이 사실상 없음 / 기하 이상 글리프 보유
           → 기하 이상은 글자만 charset 에서 빼면 그 음절이 **전 폰트 교집합**에서 사라지므로
             (UlsanJunggu '델' 선례) 폰트를 통째로 뺀다
  PARTIAL  학습 알파벳(korean_lines.txt + chars.txt 음절) 일부를 못 그림 /
           공백(U+0020) 없음 / 기본 ASCII(a-z A-Z 0-9) 결손
           → 폰트 자체는 멀쩡하지만 intersection 풀에 넣으면 교집합을 깎는다
  OK       위 어디에도 안 걸림

사용:
  .venv/bin/python tools/font_pool_audit.py --fonts-dir assets/fonts_korean_new/noonnu
  .venv/bin/python tools/font_pool_audit.py --fonts-dir <dir> --move   # 실제로 옮김
"""
from __future__ import annotations
import argparse, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from font_audit import audit_font, corpus_syllables          # noqa: E402
from font_charset import font_charset, FONT_EXTS             # noqa: E402

HERE = Path(__file__).resolve().parents[1]
ASCII_NEED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
REJECT_DIR, PARTIAL_DIR = "_rejected", "_partial"


def check(task):
    """폰트 1종 → (이름, verdict, 사유, charset). charset 은 OK/PARTIAL 일 때만 의미 있다."""
    fp, size, h_factor, w_factor, syll = task
    fp = Path(fp)
    cs, info = font_charset(fp, size, 10)
    if cs is None:
        return fp.name, "REJECT", f"폰트 로드 실패 — {info}", ""
    chars = set(cs)

    miss = [c for c in syll if c not in chars]
    if len(miss) >= len(syll) * 0.9:            # 한글이 사실상 없음(cmap 만 있고 글리프가 빈 폰트 등)
        return fp.name, "REJECT", f"한글 알파벳 {len(miss)}/{len(syll)}자를 못 그림", cs

    try:                                        # 기하 이상 — charset 으로는 못 막는 유형
        r = audit_font(fp, size, h_factor, w_factor, syll)
    except Exception as e:
        return fp.name, "REJECT", f"렌더 실패 — {type(e).__name__}: {e}", cs
    if r["broken"]:
        top = "  ".join(f"{c!r}(h×{hf},w×{wf})" for (c, w, h, hf, wf) in r["broken"][:5])
        return fp.name, "REJECT", f"기하 이상 글리프 {len(r['broken'])}자: {top}", cs

    why = []
    if miss:
        why.append(f"알파벳 {len(miss)}자 부족")
    if " " not in chars:
        why.append("공백(U+0020) 없음 → 렌더 시 .notdef 박스")
    if not ASCII_NEED <= chars:
        lo = len({c for c in "abcdefghijklmnopqrstuvwxyz"} & chars)
        up = len({c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"} & chars)
        dg = len({c for c in "0123456789"} & chars)
        why.append("영어 전무" if lo + up == 0 else f"ASCII 결손(소{lo}/26 대{up}/26 숫자{dg}/10)")
    return fp.name, ("PARTIAL" if why else "OK"), " · ".join(why), cs


def intersection_report(keep: dict, syll: list[str]) -> list[str]:
    """OK 폰트만으로 교집합이 요구사항을 만족하는지 + 상위 걸림돌."""
    if not keep:
        return ["  (OK 폰트 없음)"]
    inter = set.intersection(*(set(v) for v in keep.values()))
    hang = sum(1 for c in inter if "가" <= c <= "힣")
    need = set(syll)
    out = [f"  OK {len(keep)}종 교집합: 한글 {hang}자 · 전체 {len(inter)}자",
           f"    학습 알파벳 {len(syll)}자 확보: {'O' if need <= inter else f'X ({len(need - inter)}자 부족)'}",
           f"    공백+기본 ASCII 확보: {'O' if (ASCII_NEED | {' '}) <= inter else 'X'}"]
    if not need <= inter:                        # 누가 끌어내리는지
        drag = sorted(((len(need - set(v)), n) for n, v in keep.items() if need - set(v)), reverse=True)
        out.append("    걸림돌(부족 글자 수 순): "
                   + ", ".join(f"{n}({k})" for k, n in drag[:8]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts-dir", required=True, help="판정할 폰트 디렉토리")
    ap.add_argument("--move", action="store_true",
                    help="REJECT→_rejected/, PARTIAL→_partial/ 로 실제 이동(기본은 리포트만)")
    ap.add_argument("--size", type=int, default=100, help="기하 검사 렌더 크기(font_audit 기본과 동일)")
    ap.add_argument("--h-factor", type=float, default=2.0)
    ap.add_argument("--w-factor", type=float, default=2.5)
    ap.add_argument("--out", default=None, help="리포트 경로(기본 _val/pool_audit_<dir>.txt)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    d = Path(args.fonts_dir)
    fonts = sorted(p for p in d.iterdir() if p.suffix.lower() in FONT_EXTS)
    if not fonts:
        raise SystemExit(f"폰트가 없습니다: {d}")
    syll = corpus_syllables()
    print(f"{len(fonts)}종 판정 (학습 알파벳 {len(syll)}자)", flush=True)

    tasks = [(str(p), args.size, args.h_factor, args.w_factor, syll) for p in fonts]
    res, keep = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, verdict, why, cs) in enumerate(ex.map(check, tasks), 1):
            res.append((name, verdict, why))
            if verdict == "OK":
                keep[name] = cs
            if verdict != "OK":
                print(f"  [{i}/{len(fonts)}] [{verdict}] {name}: {why}", flush=True)

    tally = {v: sum(1 for _, x, _ in res if x == v) for v in ("OK", "PARTIAL", "REJECT")}
    lines = ["=" * 72, f"풀 판정: {d}  ({len(fonts)}종)",
             f"  OK {tally['OK']} · PARTIAL {tally['PARTIAL']} · REJECT {tally['REJECT']}", "",
             "교집합(intersection 규칙 기준)"] + intersection_report(keep, syll) + ["=" * 72, ""]
    for name, verdict, why in sorted(res, key=lambda x: (x[1] == "OK", x[0])):
        lines.append(f"{verdict:8s} {name:44s} {why}")
    out = Path(args.out) if args.out else HERE / "_val" / f"pool_audit_{d.name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"\n리포트: {out}")

    if not args.move:
        if tally["PARTIAL"] or tally["REJECT"]:
            print("(--move 를 주면 위 폰트를 _rejected/ · _partial/ 로 옮깁니다)")
        return
    for sub, verdict in ((REJECT_DIR, "REJECT"), (PARTIAL_DIR, "PARTIAL")):
        rows = [(n, w) for n, v, w in res if v == verdict]
        if not rows:
            continue
        dst = d.parent / sub
        dst.mkdir(parents=True, exist_ok=True)
        for n, _ in rows:
            if (d / n).exists():
                shutil.move(str(d / n), str(dst / n))
        rp = dst / "REASON.txt"
        prev = rp.read_text(encoding="utf-8") + "\n\n" if rp.exists() else ""
        rp.write_text(prev + f"[{verdict}] tools/font_pool_audit.py — 원본 {d}\n"
                      + "\n".join(f"{n:44s} {w}" for n, w in sorted(rows)), encoding="utf-8")
        print(f"이동 {len(rows)}종 → {dst}")
    print("⚠️ 폰트를 옮겼으면 fonts_charsets.json 을 다시 만드세요: "
          f"tools/font_charset.py --fonts-dir {d} --force")


if __name__ == "__main__":
    main()
