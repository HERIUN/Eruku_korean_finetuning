"""repo OnlineSplitFontSquare 를 한글로 사용 (온라인 style/gen split 렌더).

- style/gen 텍스트를 각각 다른 단어수 범위로 샘플 (Phase1: 둘 다 2~4 / Phase2: style 1~8, gen 1~32)
- 한글 mixed-script sampler (gen_korean_fontset.MixedLineSampler 재사용, 전역 covered=폰트 union)
- 폰트별 charset 필터(fonts_charsets.json)로 tofu 방지
- 무한 온라인 (디스크 0). 학습 샘플 저장은 dump_samples() 사용.

CLI 테스트:
  python korean_split_dataset.py --n 8 --style 1 8 --gen 2 16 --out /tmp/ksplit
"""
from __future__ import annotations
import sys, random, json, argparse
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE))
import gen_korean_fontset as G
from custom_datasets.font_square.font_square_split import OnlineSplitFontSquare
from custom_datasets.font_square import font_transforms_split as FT


class KoreanSplitFontSquare(OnlineSplitFontSquare):
    """OnlineSplitFontSquare 와 동일하되 style/gen 을 서로 다른 sampler 로 뽑음."""
    def __init__(self, fonts, backgrounds, style_sampler, gen_sampler, **kw):
        super().__init__(fonts, backgrounds, text_sampler=style_sampler, **kw)
        self.style_sampler = style_sampler
        self.gen_sampler = gen_sampler
        # RandomInvert 는 bg_patch 반전인데 합성이 곱셈(img = text*bg)이라
        # 밝은 종이 배경(≈1)이 반전되면 이미지 전체가 ≈0 (새까만 무정보 샘플).
        # 원본 font-square 의 컬러 배경에서만 유효한 증강 → 비활성화.
        for t in self.transform.transforms:
            if isinstance(t, FT.RandomInvert):
                t.p = 0.0

    def __getitem__(self, font_id):
        font_id = font_id % self.renderers_length
        style_text, gen_text = self.style_sampler(), self.gen_sampler()
        sample = self.transform({'style_text': style_text, 'gen_text': gen_text, 'font_id': font_id})
        sw = sample['style_img_width'] * sample['img'].shape[2] // sample['total_img_width']
        return {
            'style_img': sample['img'][:, :, :sw],
            'gen_img': sample['img'][:, :, sw:],
            'style_text': sample['style_text'],
            'gen_text': sample['gen_text'],
            'writer': font_id,
        }


def build_samplers(style_range, gen_range, n_english=8000, seed=42):
    rng = random.Random(seed)
    pools = G.build_pools(ASSETS / "corpus/korean_lines.txt",
                          ASSETS / "corpus/chars.txt",
                          str(ASSETS / "corpus/english_words.txt"), n_english, rng)
    cs = json.load(open(ASSETS / "fonts_korean_v2/train/fonts_charsets.json"))
    cps, inter = set(), None
    for s in cs.values():
        f = {ord(c) for c in s}
        cps |= f
        inter = f if inter is None else inter & f
    # rand 음절 풀 = 전 폰트 교집합의 한글 음절 (KS X 1001 2350자) → tofu-safe
    syls = sorted(chr(c) for c in inter if 0xAC00 <= c <= 0xD7A3)
    w = {"ko": 0.45, "en": 0.20, "num": 0.20, "rand": 0.15}
    style_s = G.MixedLineSampler(pools["ko"], pools["en"], cps, w, style_range[0], style_range[1], 40, 0.35, 0.08, rng,
                                 rand_syllables=syls)
    gen_s = G.MixedLineSampler(pools["ko"], pools["en"], cps, w, gen_range[0], gen_range[1], 130, 0.35, 0.08, rng,
                               rand_syllables=syls)
    print(f"samplers: style {style_range} gen {gen_range} | ko {len(pools['ko'])} en {len(pools['en'])} "
          f"rand_syl {len(syls)} cps {len(cps)}")
    return style_s, gen_s


def make_dataset(style_range=(1, 8), gen_range=(1, 32), length=None, seed=42):
    style_s, gen_s = build_samplers(style_range, gen_range, seed=seed)
    return KoreanSplitFontSquare(ASSETS / "fonts_korean_v2/train",
                                 str(ASSETS / "backgrounds"),
                                 style_s, gen_s, length=length)


def split_collate(batch):
    """split 데이터셋(style_img/gen_img/...) → train_korean 형식(style_img/same_img + len + text)."""
    return {
        "style_img": [b["style_img"] for b in batch],
        "same_img": [b["gen_img"] for b in batch],
        "style_text": [b["style_text"] for b in batch],
        "same_text": [b["gen_text"] for b in batch],
        "style_img_len": [b["style_img"].shape[-1] for b in batch],
        "same_img_len": [b["gen_img"].shape[-1] for b in batch],
    }


def to_gray(t):  # [3,H,W] in [-1,1] -> uint8 grayscale
    return (((t.mean(0) + 1) / 2).clamp(0, 1).numpy() * 255).astype(np.uint8)


def dump_samples(ds, n, out_dir):
    """학습에 쓰이는 형식의 샘플 n개를 저장 + montage."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    gf = ImageFont.truetype(str(ASSETS / "fonts_label/NanumGothic-Regular.ttf"), 15)
    rows = []
    for i in range(n):
        s = ds[i]
        si, gi = to_gray(s["style_img"]), to_gray(s["gen_img"])
        cv2.imwrite(str(out / f"{i:03d}_style.png"), si)
        cv2.imwrite(str(out / f"{i:03d}_gen.png"), gi)
        # montage row: label + [style | sep | gen]
        H = 64
        def fit(a):
            if a.shape[0] != H:
                a = cv2.resize(a, (max(1, int(a.shape[1] * H / a.shape[0])), H))
            return a
        si, gi = fit(si), fit(gi)
        sep = np.full((H, 6), 0, np.uint8)
        body = np.hstack([si, sep, gi])
        lab = Image.new("L", (body.shape[1], 22), 235)
        ImageDraw.Draw(lab).text((4, 3), f"[style:{s['style_text'][:24]}] | [gen:{s['gen_text'][:40]}]", font=gf, fill=0)
        rows.append(np.vstack([np.array(lab), body, np.full((4, body.shape[1]), 90, np.uint8)]))
        print(f"  {i}: style={s['style_text'][:30]!r} ({si.shape[1]}px) | gen={s['gen_text'][:40]!r} ({gi.shape[1]}px)")
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1]), 255, np.uint8)]) if r.shape[1] < W else r for r in rows]
    Image.fromarray(np.vstack(rows)).save(out / "_montage.png")
    print(f"saved {out}/_montage.png + {n} style/gen pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--style", nargs=2, type=int, default=[1, 8])
    ap.add_argument("--gen", nargs=2, type=int, default=[2, 16])
    ap.add_argument("--out", default="/tmp/ksplit")
    args = ap.parse_args()
    ds = make_dataset(style_range=tuple(args.style), gen_range=tuple(args.gen), length=10000)
    dump_samples(ds, args.n, args.out)
