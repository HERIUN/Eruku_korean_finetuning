"""Eruku 한글 학습용 오프라인 데이터셋 생성기.

폰트 = writer 1명 (person_key) 으로 매핑해서, 폰트마다 여러 개의 한글 라인
이미지를 렌더 + 증강 후 디스크에 저장하고, train_korean.py 가 그대로 읽는
lines_json (`[{image_path, text, person_key}]`) 을 만든다.

증강은 Eruku/font_square FT 파이프라인과 같은 *종류* (TPS 류 elastic warp,
small rotation, gaussian blur, 종이 배경 합성, ink alpha/jitter, stroke
dilation/erosion) 를 grayscale 단독 구현으로 미러링한다. torch/compiled-tps
의존성 없이 PIL/numpy/cv2 만 사용 (venv torch 깨져있어도 동작).

데이터 형식 (HanDBKoreanDataset 가 읽는 포맷):
  [{"image_path": "<abs png>", "text": "<한글 라인>", "person_key": "<font stem>"}]
  - 이미지: grayscale PNG, 가변 폭, load 시 64px 높이로 resize 됨
  - person_key = 폰트 식별자 → 같은 폰트의 다른 이미지가 style/same 페어가 됨

Usage:
  python gen_korean_fontset.py \
      --fonts-dir ../fonts_korean_v2 \
      --corpus ../font_ai_pipeline_work/benchmark/train_lines.json \
      --bg-dir ../font_ai_pipeline_work/bg_textrenderer \
      --out data/korean_fontset_pilot \
      --per-font-train 100 --per-font-val 30
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont, TTCollection
from PIL import Image, ImageDraw, ImageFont

HANGUL_RE = re.compile(r"[가-힣]")
FONT_EXTS = {".ttf", ".otf", ".ttc"}
EN_RE = re.compile(r"^[A-Za-z]{2,12}$")

PUNCT_TRAIL = list(".,!?…:;")                                  # 토큰 뒤 구두점
WRAP_PAIRS = [("(", ")"), ("[", "]"), ("“", "”"), ("‘", "’"), ('"', '"'), ("'", "'")]
SPECIALS = list("%&@#*+=/~$")                                  # 단독 특수기호


# ────────────────────────── corpus / text sampling ──────────────────────────
def build_pools(corpus_path: Path, extra_chars: Path | None, english_path,
                n_english: int, rng: random.Random) -> dict:
    """ko(한글 단어)+en(영어 단어) 풀. 숫자/구두점/특수기호는 런타임 생성.
    corpus_path: .txt(한 줄당 한 문장) 또는 lines_json([{text: ...}])."""
    corpus_path = Path(corpus_path)
    if corpus_path.suffix == ".txt":
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = [str(r.get("text", ""))
                 for r in json.loads(corpus_path.read_text(encoding="utf-8"))]
    ko: set[str] = set()
    for line in lines:
        for w in line.split():
            w = w.strip()
            if w and HANGUL_RE.search(w):
                ko.add(w)
    if extra_chars and Path(extra_chars).exists():
        for line in Path(extra_chars).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and HANGUL_RE.search(s):
                ko.add(s)
    en: list[str] = []
    if english_path and Path(english_path).exists():
        cand = [w.strip() for w in Path(english_path).read_text(
            encoding="utf-8", errors="ignore").splitlines()]
        cand = sorted({w for w in cand if EN_RE.match(w)})
        en = rng.sample(cand, n_english) if len(cand) > n_english else cand
    pools = {"ko": sorted(ko), "en": sorted(set(en))}
    if not pools["ko"]:
        raise RuntimeError(f"빈 ko pool — corpus 확인: {corpus_json}")
    return pools


def font_covered_words(words: list[str], cmap: dict) -> list[str]:
    """폰트 cmap 으로 모든 글자가 커버되는 단어만 (tofu 방지)."""
    return [w for w in words if all(ord(c) in cmap for c in w)]


def gen_number(rng: random.Random) -> str:
    """숫자/날짜/시간/가격/전화/퍼센트 등 다양한 포맷."""
    k = rng.randint(0, 8)
    if k == 0: return str(rng.randint(0, 9999))
    if k == 1: return f"{rng.randint(0,999)}.{rng.randint(0,99)}"
    if k == 2: return f"{rng.randint(1,100)}%"
    if k == 3: return f"{rng.randint(1,999)},{rng.randint(0,999):03d}"
    if k == 4: return f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}"
    if k == 5: return f"{rng.randint(2000,2025)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
    if k == 6: return f"010-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
    if k == 7: return f"{rng.randint(2000,2025)}년"
    return f"{rng.randint(1,9999):,}원"


class MixedLineSampler:
    """ko/en/num/rand 토큰을 가중치로 섞고 구두점·특수기호를 입힌 라인 생성.
    폰트가 커버하는 글자(covered_cps)만 사용.
    rand = 랜덤 음절 조합 가짜 단어 (논문의 'random words' 대응) — 코퍼스
    빈도 편향 없이 모든 음절을 단어 맥락으로 균등 노출."""

    def __init__(self, ko, en, covered_cps, weights, min_words, max_words,
                 max_chars, punct_prob, special_prob, rng, rand_syllables=None):
        self.ko, self.en, self.cps = ko, en, covered_cps
        self.rand = rand_syllables or []
        self.min_words, self.max_words = min_words, max_words
        self.max_chars, self.rng = max_chars, rng
        self.punct_prob, self.special_prob = punct_prob, special_prob
        avail = {"ko": ko, "en": en, "num": True, "rand": self.rand}
        self.cats = [c for c in ("ko", "en", "num", "rand")
                     if weights.get(c, 0) > 0 and avail[c]]
        self.cw = [weights[c] for c in self.cats]
        self.trail = [p for p in PUNCT_TRAIL if ord(p) in covered_cps]
        self.wraps = [(a, b) for a, b in WRAP_PAIRS
                      if ord(a) in covered_cps and ord(b) in covered_cps]
        self.specials = [s for s in SPECIALS if ord(s) in covered_cps]

    def _covered(self, s: str) -> bool:
        return bool(s) and all(ord(c) in self.cps for c in s)

    def _token(self) -> str:
        cat = self.rng.choices(self.cats, self.cw)[0]
        if cat == "ko":
            return self.rng.choice(self.ko)
        if cat == "en":
            return self.rng.choice(self.en)
        if cat == "rand":   # 랜덤 음절 1~4글자 (2~3글자 위주)
            n = self.rng.choices((1, 2, 3, 4), (0.15, 0.35, 0.30, 0.20))[0]
            return "".join(self.rng.choice(self.rand) for _ in range(n))
        for _ in range(4):
            t = gen_number(self.rng)
            if self._covered(t):
                return t
        return str(self.rng.randint(0, 999))

    def _decorate(self, tok: str) -> str:
        if self.wraps and self.rng.random() < self.punct_prob * 0.4:
            a, b = self.rng.choice(self.wraps); tok = f"{a}{tok}{b}"
        if self.trail and self.rng.random() < self.punct_prob:
            tok = tok + self.rng.choice(self.trail)
        return tok

    def __call__(self) -> str:
        n = self.rng.randint(self.min_words, self.max_words)
        toks = []
        for _ in range(n):
            t = self._decorate(self._token())
            if self._covered(t):
                toks.append(t)
            if self.specials and self.rng.random() < self.special_prob:
                toks.append(self.rng.choice(self.specials))
        txt = "".join(c for c in " ".join(toks) if c == " " or ord(c) in self.cps)
        txt = " ".join(txt.split())
        if len(txt) > self.max_chars:
            txt = txt[: self.max_chars].rstrip()
        return txt


# ───────────────────────────── rendering ─────────────────────────────
def render_text(font: ImageFont.FreeTypeFont, text: str, pad: int) -> np.ndarray:
    """검정 글씨 / 흰 배경 (255) grayscale uint8 [H, W]."""
    bbox = font.getbbox(text)
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    W, H = w + 2 * pad, h + 2 * pad
    img = Image.new("L", (W, H), 255)
    ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=0)
    return np.array(img, dtype=np.uint8)


# ───────────────────────────── augmentation ─────────────────────────────
def aug_rotate(arr: np.ndarray, max_deg: float, rng: random.Random) -> np.ndarray:
    deg = rng.uniform(-max_deg, max_deg)
    h, w = arr.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def aug_elastic(arr: np.ndarray, amp: float, sigma: float, rng: random.Random) -> np.ndarray:
    """smoothed random displacement field → 손글씨 느낌의 waviness (TPS 류 미러)."""
    h, w = arr.shape
    seed = rng.randint(0, 2**31 - 1)
    state = np.random.RandomState(seed)
    dx = cv2.GaussianBlur((state.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * amp
    dy = cv2.GaussianBlur((state.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * amp
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def aug_morph(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """ink=어두움 → erode 가 획 두껍게, dilate 가 얇게."""
    k = np.ones((2, 2), np.uint8)
    return cv2.erode(arr, k) if rng.random() < 0.5 else cv2.dilate(arr, k)


def bg_patch(bgs: list[np.ndarray], h: int, w: int, rng: random.Random) -> np.ndarray:
    """배경 grayscale 패치 (h, w). bg 없으면 흰색."""
    bg = bgs[rng.randrange(len(bgs))]
    bh, bw = bg.shape
    if bh < h or bw < w:
        bg = cv2.resize(bg, (max(w, bw), max(h, bh)), interpolation=cv2.INTER_LINEAR)
        bh, bw = bg.shape
    y0 = rng.randint(0, bh - h)
    x0 = rng.randint(0, bw - w)
    patch = bg[y0:y0 + h, x0:x0 + w].astype(np.float32)
    if rng.random() < 0.5:
        patch = patch[::-1, :]
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    return patch


def composite(ink_arr: np.ndarray, bg: np.ndarray, ink_value: float,
              alpha: float) -> np.ndarray:
    """ink(검정 글씨) 를 종이 배경에 alpha 합성."""
    mask = (255.0 - ink_arr.astype(np.float32)) / 255.0  # 1 = full ink
    eff = mask * alpha
    out = bg * (1.0 - eff) + ink_value * eff
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    b = rng.uniform(-12, 12)            # brightness
    c = rng.uniform(0.9, 1.1)           # contrast
    out = (arr.astype(np.float32) - 128) * c + 128 + b
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(ink_arr: np.ndarray, bgs: list[np.ndarray], target_h: int,
            rng: random.Random) -> np.ndarray:
    arr = ink_arr
    if rng.random() < 0.5:
        arr = aug_rotate(arr, 3.0, rng)
    if rng.random() < 0.7:
        h, w = arr.shape
        arr = aug_elastic(arr, amp=max(1.5, h * 0.04), sigma=max(4.0, w * 0.02), rng=rng)
    if rng.random() < 0.15:
        arr = aug_morph(arr, rng)
    if rng.random() < 0.4:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)

    h, w = arr.shape
    use_white = (not bgs) or rng.random() < 0.3
    bg = np.full((h, w), 255.0, np.float32) if use_white else bg_patch(bgs, h, w, rng)
    ink_value = rng.uniform(0, 55)       # 잉크 농도 (순검정 아님)
    alpha = rng.uniform(0.7, 1.0)
    out = composite(arr, bg, ink_value, alpha)

    if rng.random() < 0.5:
        out = jitter(out, rng)

    # height 64 로 resize (load 시에도 하지만 파일 크기/일관성 위해 미리)
    nh = target_h
    nw = max(1, int(round(w * (target_h / h))))
    out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


# ───────────────────────────── font loading ─────────────────────────────
def get_cmap(font_path: Path) -> dict:
    if font_path.suffix.lower() == ".ttc":
        coll = TTCollection(str(font_path))
        cmap: dict = {}
        for f in coll.fonts:
            cmap.update(f.getBestCmap() or {})
        return cmap
    return TTFont(str(font_path)).getBestCmap() or {}


def list_fonts(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in FONT_EXTS)


# ───────────────────────────── main generate ─────────────────────────────
def gen_split(fonts: list[Path], pools: dict, bgs: list[np.ndarray],
              out_img_dir: Path, per_font: int, args, rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    weights = {"ko": args.w_ko, "en": args.w_en, "num": args.w_num,
               "rand": getattr(args, "w_rand", 0.0)}
    for fi, fp in enumerate(fonts):
        person_key = fp.stem
        try:
            cmap = get_cmap(fp)
            font = ImageFont.truetype(str(fp), args.font_size)
        except Exception as e:
            print(f"  [skip] {fp.name}: 폰트 로드 실패 {e}")
            continue
        covered_cps = set(cmap.keys())
        ko_cov = font_covered_words(pools["ko"], cmap)
        en_cov = font_covered_words(pools["en"], cmap)
        if len(ko_cov) < args.min_words:
            print(f"  [skip] {fp.name}: ko 커버 {len(ko_cov)} < {args.min_words}")
            continue
        syls = sorted(chr(c) for c in covered_cps if 0xAC00 <= c <= 0xD7A3)
        sampler = MixedLineSampler(ko_cov, en_cov, covered_cps, weights,
                                   args.min_words_per_line, args.max_words_per_line,
                                   args.max_chars, args.punct_prob, args.special_prob, rng,
                                   rand_syllables=syls)
        fdir = out_img_dir / person_key
        fdir.mkdir(parents=True, exist_ok=True)
        made = 0
        attempts = 0
        while made < per_font and attempts < per_font * 5:
            attempts += 1
            text = sampler()
            if not text.strip():
                continue
            try:
                ink = render_text(font, text, pad=args.pad)
                if ink.shape[1] < 8 or ink.shape[0] < 8:
                    continue
                out = augment(ink, bgs, args.height, rng)
            except Exception as e:
                print(f"    render err {fp.name} {text!r}: {e}")
                continue
            img_path = fdir / f"{person_key}_{made:04d}.png"
            cv2.imwrite(str(img_path), out)
            rows.append({
                "id": f"{person_key}_{made:04d}",
                "text": text,
                "type": "라인(폰트합성)",
                "person_key": person_key,
                "image_path": str(img_path.resolve()),
            })
            made += 1
        print(f"  [{fi+1}/{len(fonts)}] {person_key}: {made} imgs (ko={len(ko_cov)} en={len(en_cov)})")
    return rows


def save_montage(rows: list[dict], path: Path, n: int = 12):
    if not rows:
        return
    rng = random.Random(0)
    picks = rng.sample(rows, min(n, len(rows)))
    imgs = [cv2.imread(p["image_path"], cv2.IMREAD_GRAYSCALE) for p in picks]
    W = max(im.shape[1] for im in imgs)
    canvas = []
    for im in imgs:
        pad = np.full((im.shape[0], W - im.shape[1]), 255, np.uint8)
        canvas.append(np.hstack([im, pad]))
        canvas.append(np.full((4, W), 0, np.uint8))
    cv2.imwrite(str(path), np.vstack(canvas))


def main():
    ap = argparse.ArgumentParser()
    HERE = Path(__file__).resolve().parent
    ap.add_argument("--fonts-dir", default=str(HERE / "assets" / "fonts_korean_v2"))
    ap.add_argument("--corpus", default=str(HERE / "assets" / "corpus" / "korean_lines.txt"))
    ap.add_argument("--extra-chars", default=str(HERE / "assets" / "corpus" / "chars.txt"))
    ap.add_argument("--bg-dir", default=str(HERE / "assets" / "backgrounds"))
    ap.add_argument("--out", default=str(HERE / "data" / "korean_fontset_pilot"))
    ap.add_argument("--per-font-train", type=int, default=100)
    ap.add_argument("--per-font-val", type=int, default=30)
    ap.add_argument("--font-size", type=int, default=80)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--min-words", type=int, default=50,
                    help="폰트가 이만큼 단어를 커버 못하면 skip")
    ap.add_argument("--min-words-per-line", type=int, default=1)
    ap.add_argument("--max-words-per-line", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=28)
    ap.add_argument("--english-words", default=str(HERE / "assets" / "corpus" / "english_words.txt"))
    ap.add_argument("--n-english", type=int, default=8000)
    ap.add_argument("--w-ko", type=float, default=0.55, help="라인 토큰 중 한글 비중")
    ap.add_argument("--w-en", type=float, default=0.22, help="영어 비중")
    ap.add_argument("--w-num", type=float, default=0.23, help="숫자 비중")
    ap.add_argument("--w-rand", type=float, default=0.0,
                    help="랜덤 음절 가짜단어 비중 (오프라인 ref set 은 기본 0; 온라인 학습은 0.15)")
    ap.add_argument("--punct-prob", type=float, default=0.35, help="토큰에 구두점 붙일 확률")
    ap.add_argument("--special-prob", type=float, default=0.08, help="특수기호 단독 삽입 확률")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts_dir = Path(args.fonts_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 토큰 풀 (ko + en); 숫자/구두점/특수기호는 런타임 생성
    pools = build_pools(Path(args.corpus), Path(args.extra_chars),
                        args.english_words, args.n_english, rng)
    print(f"pools: ko={len(pools['ko'])} en={len(pools['en'])} "
          f"| weights ko/en/num={args.w_ko}/{args.w_en}/{args.w_num} punct={args.punct_prob}")

    # backgrounds (grayscale)
    bgs: list[np.ndarray] = []
    bg_dir = Path(args.bg_dir)
    if bg_dir.is_dir():
        for p in sorted(bg_dir.iterdir()):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    bgs.append(im)
    print(f"backgrounds: {len(bgs)}")

    # fonts: train/ val/ 하위 디렉토리 우선, 없으면 flat
    train_fonts_dir = fonts_dir / "train"
    val_fonts_dir = fonts_dir / "val"
    if train_fonts_dir.is_dir():
        train_fonts = list_fonts(train_fonts_dir)
        val_fonts = list_fonts(val_fonts_dir) if val_fonts_dir.is_dir() else []
    else:
        train_fonts = list_fonts(fonts_dir)
        val_fonts = []
    print(f"fonts: train={len(train_fonts)} val={len(val_fonts)}")

    print("== TRAIN ==")
    train_rows = gen_split(train_fonts, pools, bgs, out_dir / "images_train",
                           args.per_font_train, args, rng)
    print("== VAL ==")
    val_rows = gen_split(val_fonts, pools, bgs, out_dir / "images_val",
                         args.per_font_val, args, rng) if val_fonts else []

    train_json = out_dir / "train_lines.json"
    train_json.write_text(json.dumps(train_rows, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"\nwrote {train_json}  ({len(train_rows)} rows, "
          f"{len({r['person_key'] for r in train_rows})} writers)")
    if val_rows:
        val_json = out_dir / "val_lines.json"
        val_json.write_text(json.dumps(val_rows, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"wrote {val_json}  ({len(val_rows)} rows)")

    save_montage(train_rows, out_dir / "sample_montage.png")
    print(f"montage: {out_dir / 'sample_montage.png'}")


if __name__ == "__main__":
    main()
