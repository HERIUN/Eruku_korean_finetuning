"""폰트별 **실렌더 기준** charset 생성 (`fonts_charsets.json` 호환).

`custom_datasets/korean/split.py:ensure_fonts_charsets` 는 charset 을 **cmap 만 보고** 만든다
(`chr(c) for c in get_cmap(p)`). cmap 은 "이 codepoint 를 그릴 수 있다"는 폰트의 *주장*일 뿐이라
두 가지를 못 거른다 — 둘 다 렌더러의 charset 필터(`render_font.py:75`)를 그대로 통과한다:

  ① 빈 글리프  cmap 엔 있는데 잉크가 없다 → 이미지엔 안 그려지는데 라벨엔 글자가 남는다
                (= **이미지-라벨 불일치**. 두부보다 위험하다)
  ② placeholder(두부)  한 글리프를 수백~수천 codepoint 에 alias 해 □ ▨ ⊠ 로 그린다

이게 docs/DATA_LIMITATIONS.md D6 이고 cmap 기반으로는 원리상 못 고친다. 여기서는 실제로 래스터해서
잉크가 있는 글자만 남기고, 아래 둘 중 하나면 placeholder 로 보고 뺀다:

  ②-a **.notdef 와 같은 비트맵** — cmap 에 없는 codepoint 를 그려 폰트의 .notdef(□) 를 확보한 뒤
       대조한다. 몇 자가 공유하든 무관하므로 **박스를 소수만 쓰는 폰트도 잡는다**
  ②-b **대량 alias** — 같은 비트맵을 `--tofu-min`(기본 10)자 이상이 공유. .notdef 가 아닌 별도
       박스 글리프를 쓰는 폰트를 잡는다(KBIZHanmaumMyungjo 의 한자·기호 1,555자가 이 경우)

**ASCII 인쇄가능 문자(0x20~0x7E)는 두부 판정에서 제외**한다. 악센트 변형(À Á Â …)이 민짜 `A` 로
그려지는 폰트에서 비트맵 공유가 임계값을 넘는데, 이때 잘못 그려진 쪽은 À 지 `A` 가 아니다.
(실측: GabiaSai-Regular 에서 A E I O U a e i 가 오탐으로 잡혔다)

한계: 글자마다 **다르게** 깨진 글리프는 비트맵이 서로 달라 못 잡는다 → `tools/font_audit.py` 의
기하 검사(중앙값 대비 높이/폭 이상치)가 그 영역을 맡는다.

사용:
  .venv/bin/python tools/font_charset.py --fonts-dir assets/fonts_korean_new/ownglyph
  .venv/bin/python tools/font_charset.py --fonts-dir assets/fonts_korean_v2/train --out _val/x.json
"""
from __future__ import annotations
import argparse, json, unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import ImageFont
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
FONT_EXTS = (".ttf", ".otf")
ASCII_KEEP = {c for c in range(0x20, 0x7F)}  # 두부 판정 면제(악센트 오탐 방지)
# cmap 에 없을 가능성이 매우 높은 codepoint — 이걸 그리면 폰트의 .notdef 글리프가 나온다
NOTDEF_PROBES = (0x0FFF8, 0xE0FFF, 0x10FFFD, 0x2FFFD, 0x0DFFF)


def notdef_hashes(cmap, pil):
    """폰트의 .notdef(□) 비트맵 해시 집합. 잉크가 없으면(빈 .notdef) 빈 집합."""
    out = set()
    for cp in NOTDEF_PROBES:
        if cp in cmap:
            continue
        try:
            m = pil.getmask(chr(cp))
        except Exception:
            continue
        if m.size[0] > 0 and m.getbbox() is not None:
            out.add(hash((m.size, bytes(m))))
    return out


def font_charset(fp: Path, size: int, tofu_min: int):
    """폰트 1종 → (charset 문자열, 통계). 실패 시 (None, 사유)."""
    try:
        cmap = TTFont(str(fp), fontNumber=0, lazy=True).getBestCmap()
        pil = ImageFont.truetype(str(fp), size)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    nd = notdef_hashes(cmap, pil)          # ②-a: .notdef 와 같으면 몇 자든 두부
    keep, inked, hits, empty = [], [], {}, 0
    for cp in cmap:
        ch = chr(cp)
        if unicodedata.category(ch)[0] in ("Z", "C"):
            keep.append(ch)                  # 공백·제어는 잉크 없는 게 정상
            continue
        try:
            m = pil.getmask(ch)
            if m.size[0] <= 0 or m.getbbox() is None:
                empty += 1                   # ① 빈 글리프
                continue
            h = hash((m.size, bytes(m)))
        except Exception:
            empty += 1
            continue
        inked.append((cp, ch, h))
        hits[h] = hits.get(h, 0) + 1

    shared = {h for h, k in hits.items() if k >= tofu_min}
    tofu = notdef_hit = 0
    for cp, ch, h in inked:
        if h in nd:                          # ②-a .notdef 로 그려짐 (공유 수 무관)
            notdef_hit += 1
            continue
        if h in shared and cp not in ASCII_KEEP:
            tofu += 1                        # ②-b 대량 alias placeholder
            continue
        keep.append(ch)
    return "".join(sorted(keep)), {"cmap": len(cmap), "kept": len(keep), "empty": empty,
                                   "tofu": tofu, "notdef": notdef_hit}


def _work(task):
    fp, size, tofu_min = task
    cs, info = font_charset(Path(fp), size, tofu_min)
    return Path(fp).name, cs, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts-dir", nargs="+", required=True, help="폰트 디렉토리(복수 가능)")
    ap.add_argument("--out", default=None,
                    help="출력 json (기본: 각 fonts-dir 아래 fonts_charsets.json). "
                         "여러 디렉토리를 한 파일로 합칠 때만 지정")
    ap.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기 허용")
    ap.add_argument("--size", type=int, default=48, help="래스터 크기(px)")
    ap.add_argument("--tofu-min", type=int, default=10,
                    help="같은 비트맵을 이 수 이상 codepoint 가 공유하면 placeholder 로 간주")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.fonts_dir]
    fonts = [p for d in dirs for p in sorted(d.iterdir()) if p.suffix.lower() in FONT_EXTS]
    if not fonts:
        raise SystemExit(f"폰트가 없습니다: {args.fonts_dir}")
    if args.out is None and len(dirs) > 1:
        raise SystemExit("디렉토리가 여럿이면 --out 으로 합칠 파일을 지정하세요.")
    outs = [Path(args.out)] if args.out else [d / "fonts_charsets.json" for d in dirs]
    for o in outs:
        if o.exists() and not args.force:
            raise SystemExit(f"이미 있습니다(덮어쓰려면 --force): {o}")

    print(f"{len(fonts)}종 실렌더 charset 생성 (size={args.size}, tofu-min={args.tofu_min})",
          flush=True)
    per_dir: dict[Path, dict] = {d: {} for d in dirs}
    by_name = {p.name: p.parent for p in fonts}
    tot = {"empty": 0, "tofu": 0, "notdef": 0}
    failed = []
    tasks = [(str(p), args.size, args.tofu_min) for p in fonts]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, cs, info) in enumerate(ex.map(_work, tasks), 1):
            if cs is None:
                failed.append(f"{name}: {info}")
                print(f"  [{i}/{len(fonts)}] [FAIL] {name}: {info}", flush=True)
                continue
            per_dir[by_name[name]][name] = cs
            for k in tot:
                tot[k] += info[k]
            if info["empty"] or info["tofu"] or info["notdef"]:
                print(f"  [{i}/{len(fonts)}] {name}: cmap {info['cmap']:,} → {info['kept']:,} "
                      f"(빈 {info['empty']:,} · 두부 {info['tofu']:,} · notdef {info['notdef']:,})",
                      flush=True)

    if args.out:
        merged = {n: c for d in dirs for n, c in per_dir[d].items()}
        Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"\n{len(merged)}종 → {args.out}")
    else:
        for d in dirs:
            p = d / "fonts_charsets.json"
            p.write_text(json.dumps(per_dir[d], ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"\n{len(per_dir[d])}종 → {p}")
    print(f"제외 글자: 빈 글리프 {tot['empty']:,} · placeholder {tot['tofu']:,} "
          f"· .notdef {tot['notdef']:,}")
    if failed:
        print(f"⚠️ 실패 {len(failed)}종: " + "; ".join(failed[:5]))


if __name__ == "__main__":
    main()
