"""HuggingFace 릴리즈(공개 API)로 생성 — 로컬 체크포인트 없이 도는 경로.

`infer/show.py` 는 학습 클래스 + 로컬 `.pth` 를 쓴다. 그 `.pth` 는 저장소에 없으므로
(`finetune_runs/` 는 gitignore) 새로 clone 한 환경에선 못 돈다. 이 스크립트는 대신
발행된 repo 두 개를 받아 돌린다 — T5+proj 는 `--repo`, VAE 가중치는 그 config 의
`vae_name_or_path` 가 가리키는 repo 에서 **remote code 가 알아서** 내려받는다.

⚠️ 릴리즈 클래스는 style prefix 를 `generate_handwriting()` **안에서** 잘라낸다(학습
클래스는 안 자르고 호출자가 자른다). 그래서 여기서 `infer.show.crop_gen` 을 겹쳐 쓰면
이중 크롭으로 생성물이 날아간다 — 릴리즈 경로는 `generate_handwriting` 하나로만 쓴다.

사용:
  python infer/release.py                                   # 기본 문장으로 그리드
  python infer/release.py --seed-text "쓰고 싶은 문장" --cfgs 1.0 1.5
  python infer/release.py --style-image my_handwriting.png --bbox-crop   # 실제 스캔 style
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))
# remote code 로더는 HF_MODULES_CACHE 아래로 모듈을 복사해 import 한다. 기본 경로가
# 다른 사용자 소유면 PermissionError 가 나므로 transformers import 전에 고정한다.
os.environ.setdefault("HF_MODULES_CACHE", f"/tmp/hf_modules_{os.getuid()}")

from infer.show import cell, label_img, render_in_font, FONTS_DIR   # noqa: E402

DEF_REPO = "HERIUN/eruku_korean"


def load_release(repo: str, device: str):
    """발행 repo → 릴리즈 모델. VAE 는 config.vae_name_or_path 의 repo 에서 자동 로드."""
    import torch
    from transformers import AutoModel
    print(f"[release] {repo} 로드 (VAE 는 config 가 가리키는 repo 에서 자동)")
    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    print(f"[release] vae_name_or_path = {model.config.vae_name_or_path}")
    return model.to(torch.device(device)).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEF_REPO, help="발행된 상위 모델 repo")
    ap.add_argument("--lines-json", default=str(HERE / "data/ref_set_clean/train_lines.json"),
                    help="style ref 세트. 없으면 --style-image 를 쓸 것")
    ap.add_argument("--style-image", default=None,
                    help="style 이미지 1장 직접 지정(lines-json 대신). 실제 손글씨면 --bbox-crop 권장")
    ap.add_argument("--style-text", default="", help="--style-image 의 전사(있으면 정확도↑)")
    ap.add_argument("--seed-text", default="오늘 날씨가 좋아서 친구와 공원에서 커피를 마셨다 2024년")
    ap.add_argument("--cfgs", nargs="+", type=float, default=[1.0, 1.5])
    ap.add_argument("--n-writers", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--bbox-crop", action="store_true",
                    help="style 이미지를 잉크 bbox 로 크롭(실제 스캔 손글씨의 여백 제거). 폰트 렌더엔 불필요")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-w", type=int, default=340)
    ap.add_argument("--cell-h", type=int, default=72)
    ap.add_argument("--exclude-writers", nargs="*", default=["UlsanJunggu"])
    ap.add_argument("--out", default="_debug/release_show.png")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_release(args.repo, args.device)
    CW, CH = args.cell_w, args.cell_h
    rng = random.Random(args.seed)

    # style 후보: --style-image 1장, 또는 lines_json 에서 writer 추첨
    if args.style_image:
        refs = [{"person_key": Path(args.style_image).stem, "image_path": args.style_image,
                 "text": args.style_text}]
    else:
        lj = Path(args.lines_json)
        if not lj.exists():
            raise SystemExit(f"style ref 세트가 없다: {lj}\n"
                             "  → ./inference.sh refset --per-font-train 5 --per-font-val 0 "
                             "--no-augment --out data/ref_set_clean\n"
                             "  (또는 --style-image 로 이미지 1장을 직접 지정)")
        by_w = {}
        for r in json.loads(lj.read_text(encoding="utf-8")):
            by_w.setdefault(r["person_key"], []).append(r)
        excl = set(args.exclude_writers or [])
        by_w = {w: v for w, v in by_w.items() if w.split("#")[0] not in excl}
        refs = [rng.choice(by_w[w]) for w in rng.sample(list(by_w), min(args.n_writers, len(by_w)))]

    grid, full_w = [], (2 + len(args.cfgs)) * (CW + 4)
    for ref in refs:
        w = ref["person_key"]
        style_arr = np.array(Image.open(ref["image_path"]).convert("L"))
        base = w.split("#")[0]
        fp = Path(ref["font_path"]) if ref.get("font_path") else FONTS_DIR / f"{base}.ttf"
        gt_arr = render_in_font(fp, args.seed_text) if fp.exists() else np.full((64, 200), 230, np.uint8)

        cells = [cell(style_arr, f"스타일 ref: {w}", CW, CH),
                 cell(gt_arr, f"정답 폰트: {base}", CW, CH, highlight=True)]
        for c in args.cfgs:
            out = model.generate_handwriting(
                style_image=Image.fromarray(style_arr), gen_text=args.seed_text,
                style_text=ref.get("text", ""), cfg_scale=c,
                max_new_tokens=args.max_new_tokens, bbox_crop=args.bbox_crop)
            cells.append(cell(np.array(out.convert("L")), f"생성 cfg={c}", CW, CH))
        row = np.hstack(cells)
        if row.shape[1] < full_w:
            row = np.hstack([row, np.full((row.shape[0], full_w - row.shape[1]), 255, np.uint8)])
        grid += [row, np.full((6, row.shape[1]), 80, np.uint8)]
        print(f"  {w}: ref={ref.get('text', '')[:30]!r}")

    body = np.vstack(grid)
    W = body.shape[1]
    title = label_img(f'목표 텍스트:  "{args.seed_text}"      (HF 릴리즈 {args.repo})',
                      W, 46, 22, bold=True, center=False, bg=255)
    note = label_img('← 조건(스타일) | 이 폰트로 쓴 정답모양 | 릴리즈 API 가 생성한 것(cfg별) →',
                     W, 28, 14, center=False, bg=255)
    out = np.vstack([title, np.full((2, W), 0, np.uint8), note, np.full((3, W), 0, np.uint8), body])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
