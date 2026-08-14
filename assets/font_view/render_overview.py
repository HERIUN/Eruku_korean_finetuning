"""assets 폰트 오버뷰: 폴더별로 모든 폰트를 렌더한 montage PNG 생성 (assets/font_view/*.png).

각 행 = [파일이름 + 지원 글자수 분류(전체/한글/한자/영문/숫자/기호/기타) | 샘플 1줄 렌더
(한글+영어 대/소문자+숫자+특수기호 포함, 칼럼 폭 초과 시 자동 축소)].
어떤 글씨체인지, 한글/라틴을 실제로 지원하는지 한눈에 확인용. 로드 실패 폰트는 행에 표시.
분류 기준: 한글=음절(가~힣)+자모, 한자=CJK 통합(확장 포함), 영문=A-Za-z, 숫자=0-9,
기호=유니코드 카테고리 P(문장부호)/S(기호), 기타=나머지(타 문자권·공백·제어 등).
글자수는 cmap 매핑 중 **실제 글리프가 유의미한 것만** 유효로 집계. cmap 만 믿으면 가짜 지원이
잡히므로 두 가지를 걸러 별도 표기:
  - 빈글리프: 렌더 시 잉크가 전혀 없음 (mask bbox 기준, tools.font_audit 과 동일)
  - tofu: 잉크는 있지만 **동일 비트맵이 TOFU_MIN(10)자 이상에 재사용**되는 placeholder
    (두부 박스는 보통 한 글리프를 수백~수천 codepoint 에 alias → 비트맵 해시로 탐지.
     대소문자 alias 같은 정상 재사용은 글리프당 2~4자라 threshold 미만)
(공백·제어문자는 원래 잉크가 없으므로 검사 없이 '기타'로 집계. 잔여 한계: 글자마다 다른
모양으로 깨진 글리프는 못 잡음 → 기하 이상 검사는 tools/font_audit.py 사용)

사용:
  .venv/bin/python assets/font_view/render_overview.py            # 전 폴더 → assets/font_view/*.png
  .venv/bin/python assets/font_view/render_overview.py --dirs assets/fonts_korean_v2/test
"""
from __future__ import annotations
import argparse
import unicodedata
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent          # assets/font_view
ASSETS = HERE.parent
LABEL_FONT = ASSETS / "fonts_label" / "NanumGothic-Regular.ttf"

# 한 줄에 한글 + 영어 대/소문자 + 숫자 + 특수기호 모두 포함
SAMPLE = "다람쥐 헌 쳇바퀴 Quick Fox 0123 ?!@#%"

DEFAULT_DIRS = [
    ASSETS / "fonts_korean_v2" / "train",
    ASSETS / "fonts_korean_v2" / "test",
    ASSETS / "custom_hw_fonts",
    ASSETS / "fonts_english",
    ASSETS / "fonts_label",
]

ROW_H = 116         # 행 높이(px): 샘플 1줄(라벨 3줄 기준)
NAME_W = 400        # 이름/글자수 칼럼 폭
SAMPLE_W = 1150     # 샘플 칼럼 폭
FONT_SIZE = 46
PAD = 10


TOFU_MIN = 10   # 동일 비트맵을 이 수 이상 codepoint 가 공유하면 placeholder(tofu) 의심


def charset_counts(fp: Path):
    """cmap 중 실제 글리프가 유의미한 글자를 {한글, 한자, 영문, 숫자, 기호, 기타} 로 분류.

    1-pass: 각 글자를 mask 렌더 → 잉크 없으면 empty, 있으면 비트맵 해시 수집.
    2-pass: 같은 해시가 TOFU_MIN 이상 재사용된 글자는 tofu(있는 척 placeholder)로 분리,
            나머지만 valid 로 분류. 공백·제어문자(카테고리 Z/C)는 검사 없이 '기타'.
    반환 dict: cmap, valid, empty, tofu, 분류별 개수. 파싱 실패 시 None."""
    try:
        cmap = TTFont(str(fp), lazy=True).getBestCmap()
        pil = ImageFont.truetype(str(fp), 48)
    except Exception:
        return None
    n = dict(cmap=len(cmap), valid=0, empty=0, tofu=0,
             hangul=0, hanja=0, latin=0, digit=0, symbol=0, etc=0)
    entries, hcount = [], {}                 # (cp, cat, tag=ws|empty|hash)
    for c in cmap:
        cat = unicodedata.category(chr(c))
        if cat[0] in ("Z", "C"):             # 공백/제어: 잉크 없는 게 정상
            entries.append((c, cat, "ws"))
            continue
        try:
            m = pil.getmask(chr(c))
            if m.size[0] <= 0 or m.size[1] <= 0 or m.getbbox() is None:
                entries.append((c, cat, "empty"))
                continue
            h = hash((m.size, bytes(m)))
        except Exception:
            entries.append((c, cat, "empty"))
            continue
        entries.append((c, cat, h))
        hcount[h] = hcount.get(h, 0) + 1
    tofu_hashes = {h for h, k in hcount.items() if k >= TOFU_MIN}
    for c, cat, tag in entries:
        if tag == "empty":
            n["empty"] += 1
            continue
        if tag != "ws" and tag in tofu_hashes:
            n["tofu"] += 1                   # 잉크는 있지만 동일 비트맵 대량 재사용 = 두부
            continue
        n["valid"] += 1
        if 0xAC00 <= c <= 0xD7A3 or 0x1100 <= c <= 0x11FF or 0x3130 <= c <= 0x318F:
            n["hangul"] += 1        # 음절 + 자모/호환자모
        elif 0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or 0xF900 <= c <= 0xFAFF or 0x20000 <= c <= 0x2FA1F:
            n["hanja"] += 1         # CJK 통합/확장/호환 한자
        elif 0x41 <= c <= 0x5A or 0x61 <= c <= 0x7A:
            n["latin"] += 1         # A-Z a-z
        elif 0x30 <= c <= 0x39:
            n["digit"] += 1         # 0-9
        elif cat[0] in ("P", "S"):
            n["symbol"] += 1        # 문장부호(P)·기호(S)
        else:
            n["etc"] += 1           # 타 문자권 등
    return n


def render_row(fp: Path, label_font, label_small) -> Image.Image:
    row = Image.new("RGB", (NAME_W + SAMPLE_W, ROW_H), "white")
    d = ImageDraw.Draw(row)
    # 이름 칼럼: 파일이름 + 지원 글자수 분류(2줄)
    d.text((PAD, ROW_H // 2 - 34), fp.name, font=label_font, fill="black", anchor="lm")
    n = charset_counts(fp)
    if n is None:
        d.text((PAD, ROW_H // 2 + 2), "글자수 파싱 실패", font=label_small, fill=(200, 0, 0), anchor="lm")
    else:
        bad = ""
        if n["empty"]:
            bad += f" · 빈 {n['empty']:,}"
        if n["tofu"]:
            bad += f" · tofu {n['tofu']:,}"
        d.text((PAD, ROW_H // 2 + 2),
               f"유효 {n['valid']:,}/{n['cmap']:,} · 한글 {n['hangul']:,} · 한자 {n['hanja']:,}",
               font=label_small, fill=(90, 90, 90), anchor="lm")
        d.text((PAD, ROW_H // 2 + 28),
               f"영문 {n['latin']} · 숫자 {n['digit']} · 기호 {n['symbol']:,} · 기타 {n['etc']:,}" + bad,
               font=label_small, fill=(150, 60, 60) if bad else (90, 90, 90), anchor="lm")
    # 샘플 칼럼: 통합 샘플 1줄 (폭 초과 시 폰트 크기 자동 축소)
    try:
        f = ImageFont.truetype(str(fp), FONT_SIZE)
        w = d.textlength(SAMPLE, font=f)
        if w > SAMPLE_W - 2 * PAD:
            f = ImageFont.truetype(str(fp), max(16, int(FONT_SIZE * (SAMPLE_W - 2 * PAD) / w)))
        d.text((NAME_W + PAD, ROW_H // 2), SAMPLE, font=f, fill="black", anchor="lm")
    except Exception as e:
        d.text((NAME_W + PAD, ROW_H // 2), f"[로드 실패: {type(e).__name__}]",
               font=label_font, fill="red", anchor="lm")
    d.line([(0, ROW_H - 1), (NAME_W + SAMPLE_W, ROW_H - 1)], fill=(220, 220, 220))
    d.line([(NAME_W - 1, 0), (NAME_W - 1, ROW_H)], fill=(220, 220, 220))
    return row


def render_dir(d: Path, out_dir: Path, label_font, label_small) -> Path | None:
    fonts = sorted(p for p in d.glob("*") if p.suffix.lower() in (".ttf", ".otf"))
    if not fonts:
        print(f"skip(폰트 없음): {d}")
        return None
    header_h = 40
    canvas = Image.new("RGB", (NAME_W + SAMPLE_W, header_h + ROW_H * len(fonts)), "white")
    dr = ImageDraw.Draw(canvas)
    title = f"{d.relative_to(ASSETS)}  ({len(fonts)} fonts)"
    dr.rectangle([0, 0, canvas.width, header_h], fill=(40, 40, 40))
    dr.text((PAD, header_h // 2), title, font=label_font, fill="white", anchor="lm")
    for i, fp in enumerate(fonts):
        canvas.paste(render_row(fp, label_font, label_small), (0, header_h + i * ROW_H))
    name = "overview_" + str(d.relative_to(ASSETS)).replace("/", "_") + ".png"
    out = out_dir / name
    canvas.save(out)
    print(f"{out}  ({len(fonts)} fonts)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="스캔할 폰트 디렉토리(미지정 시 assets 기본 폴더 전체)")
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label_font = ImageFont.truetype(str(LABEL_FONT), 22)
    label_small = ImageFont.truetype(str(LABEL_FONT), 18)
    targets = [Path(d) for d in args.dirs] if args.dirs else DEFAULT_DIRS
    for d in targets:
        if d.is_dir():
            render_dir(d, out_dir, label_font, label_small)
        else:
            print(f"skip(디렉토리 아님): {d}")


if __name__ == "__main__":
    main()
