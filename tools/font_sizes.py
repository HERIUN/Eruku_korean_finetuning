"""폰트별 캘리브레이션 크기 캐시(`fonts_sizes.json`)를 만든다 — docs/DATA_LIMITATIONS.md D7.

캐시가 없으면 `font_square_split.make_renderers` 가 폰트마다 `Render.calibrate()` 를 돈다.
그것도 **순차 루프**라(`font_square_split.py:151`) 폰트 12,000종이면 학습 기동이 사실상 막힌다.
같은 계산을 프로세스 풀로 한 번만 해서 폰트 디렉토리에 덤프해두면 이후 실행은 json 만 읽는다.

파라미터는 `OnlineSplitFontSquare` 가 부르는 값과 **동일해야** 한다
(`font_square_split.py:167` calib_threshold=0.8, 나머지 기본값). calibrate 는 charset 을
안 보므로 charset 캐시와 독립이다.

사용:
  .venv/bin/python tools/font_sizes.py --fonts-dir assets/fonts_korean_v3/train
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from custom_datasets.upstream.font_square.render_font import Render  # noqa: E402

FONT_EXTS = (".ttf", ".otf", ".ttc")


def calib(fp: str):
    try:
        return Path(fp).name, Render(fp, None, None, 64, None).calibrate(None, 0.8, 128)
    except Exception as e:
        print(f"  [SKIP] {Path(fp).name}: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    d = Path(a.fonts_dir)
    fonts = sorted(str(p) for p in d.iterdir() if p.suffix.lower() in FONT_EXTS)
    print(f"{d}: 폰트 {len(fonts)}종 캘리브레이션 (workers {a.workers}) …", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        sizes = dict(r for r in ex.map(calib, fonts, chunksize=8) if r)
    out = d / "fonts_sizes.json"
    out.write_text(json.dumps(sizes, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out} — {len(sizes)}종 (실패 {len(fonts) - len(sizes)})")


if __name__ == "__main__":
    main()
