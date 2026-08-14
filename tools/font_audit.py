"""학습 폰트 데이터 품질 감사.

83개 학습 폰트 × 현대 한글 음절 11172자(가~힣)를 스캔해:
  1) 미지원 글리프(cmap 없음 = 렌더 시 두부/공백) 개수
  2) 비정상 기하 글리프 = 폰트 자체 중앙값 대비 높이/폭이 과도(예: UlsanJunggu '델' h=693 vs 중앙값 ~100)
     → 받침 분리·컴포넌트 오배치 등 malformed glyph.

per-font 중앙값 기준(폰트마다 자연 크기 다름 → 장식폰트 일괄 오탐 방지).
CPU only(getbbox·cmap), GPU/학습 무관.

사용: .venv/bin/python tools/font_audit.py [--fonts-dir ...] [--h-factor 2.0] [--out _val/font_audit.txt]
"""
from __future__ import annotations
import argparse, statistics
from pathlib import Path

from PIL import ImageFont
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
ALL_SYLL = [chr(cp) for cp in range(0xAC00, 0xD7A4)]   # 11172 완성형 음절


def corpus_syllables():
    """학습 코퍼스에 실제 등장하는 한글 음절만(= 학습이 렌더하는 대상). 없으면 전체 11172."""
    txt = ""
    for p in ("assets/corpus/korean_lines.txt", "assets/corpus/chars.txt"):
        fp = HERE / p
        if fp.exists():
            txt += fp.read_text(encoding="utf-8", errors="ignore")
    syl = sorted(c for c in set(txt) if "가" <= c <= "힣")
    return syl if syl else ALL_SYLL


def supported_codepoints(path):
    """cmap 에 실제 매핑된 codepoint 집합(=폰트가 그릴 수 있는 글자)."""
    try:
        t = TTFont(str(path), fontNumber=0, lazy=True)
        cmap = t.getBestCmap()
        t.close()
        return set(cmap.keys())
    except Exception as e:
        return None  # 못 읽으면 None → 지원검사 skip


def audit_font(path, size, h_factor, w_factor, syll):
    f = ImageFont.truetype(str(path), size)
    sup = supported_codepoints(path)
    heights, widths, geo = [], [], []
    unsupported = []
    for c in syll:
        if sup is not None and ord(c) not in sup:
            unsupported.append(c)
            continue
        b = f.getbbox(c)
        w, h = b[2] - b[0], b[3] - b[1]
        if w <= 0 or h <= 0:            # cmap 은 있지만 빈 글리프
            unsupported.append(c)
            continue
        heights.append(h); widths.append(w); geo.append((c, w, h))
    if not heights:
        return {"unsupported": len(unsupported), "total": len(syll), "unsup_list": unsupported,
                "med_h": 0, "med_w": 0, "broken": [], "all_unsupported": True}
    med_h, med_w = statistics.median(heights), statistics.median(widths)
    broken = [(c, w, h, round(h / med_h, 1), round(w / med_w, 1))
              for (c, w, h) in geo
              if h > med_h * h_factor or w > med_w * w_factor]
    broken.sort(key=lambda x: -max(x[3], x[4]))
    return {"unsupported": len(unsupported), "total": len(syll), "unsup_list": unsupported,
            "med_h": med_h, "med_w": med_w, "broken": broken, "all_unsupported": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--h-factor", type=float, default=2.0, help="높이 중앙값 대비 이 배수 초과 = 비정상")
    ap.add_argument("--w-factor", type=float, default=2.5, help="폭 중앙값 대비 이 배수 초과 = 비정상")
    ap.add_argument("--out", default=str(HERE / "_val/font_audit.txt"))
    ap.add_argument("--all-syllables", action="store_true",
                    help="코퍼스 대신 전체 11172자로 검사(대부분 폰트가 상용 2350만 담아 오탐 많음)")
    args = ap.parse_args()

    syll = ALL_SYLL if args.all_syllables else corpus_syllables()
    scope = "전체 11172자" if args.all_syllables else f"학습 코퍼스 {len(syll)}자"

    fonts = sorted(Path(args.fonts_dir).glob("*.ttf")) + sorted(Path(args.fonts_dir).glob("*.otf"))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# 폰트 품질 감사: {args.fonts_dir}  ({len(fonts)}개 폰트 × {scope})",
             f"# 판정: 미지원(cmap 없음/빈글리프 = 학습 시 두부), "
             f"비정상기하(높이>중앙값×{args.h_factor} 또는 폭>중앙값×{args.w_factor} = malformed)", ""]
    n_clean = n_broken = n_missing = 0
    problem_fonts, missing_fonts = [], []

    for fp in fonts:
        r = audit_font(fp, args.size, args.h_factor, args.w_factor, syll)
        miss = r["unsupported"]
        cov_pct = 100 * (r["total"] - miss) / r["total"]
        nbr = len(r["broken"])
        flag = ""
        if miss > 0:                       # 코퍼스 글자를 못 그림 = 학습 두부 발생
            flag = f"  ★ 코퍼스 {miss}자 미지원(두부)"; n_missing += 1
            missing_fonts.append((fp.name, miss))
        if nbr:
            n_broken += 1; problem_fonts.append((fp.name, nbr))
        if not nbr and miss == 0:
            n_clean += 1
        lines.append(f"{fp.name:38s} 코퍼스cov {cov_pct:5.1f}% ({miss:4d} 미지원)  비정상 {nbr}개{flag}")
        if miss and miss <= 40:
            lines.append("      미지원: " + " ".join(r["unsup_list"][:40]))
        if nbr:
            lines.append("      비정상: " + "  ".join(
                f"{c!r}(h×{hf},w×{wf})" for (c, w, h, hf, wf) in r["broken"][:25]))
            if nbr > 25:
                lines.append(f"      ... 외 {nbr-25}개")

    summary = ["=" * 72,
               f"요약: 폰트 {len(fonts)}개 (검사범위 {scope})",
               f"  깨끗(코퍼스 100%·비정상0): {n_clean}",
               f"  코퍼스 글자 미지원(→학습 두부): {n_missing}  "
               + (", ".join(f"{n}({m}자)" for n, m in sorted(missing_fonts, key=lambda x: -x[1])) if missing_fonts else "없음"),
               f"  비정상 기하 글리프: {n_broken}  "
               + (", ".join(f"{n}({k})" for n, k in sorted(problem_fonts, key=lambda x: -x[1])) if problem_fonts else "없음"),
               "=" * 72]
    out.write_text("\n".join(summary + [""] + lines), encoding="utf-8")
    print("\n".join(summary))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
