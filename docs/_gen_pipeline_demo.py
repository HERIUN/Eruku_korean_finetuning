"""원본 OnlineSplitFontSquare 파이프라인을 한 단계씩 실행하며 중간 이미지를 저장.
회전/워핑이 왜 '폭으로 자르기'를 어긋나게 하는지 시각적으로 보여주기 위해
회전/워핑 확률을 1.0으로 강제하고 각도도 크게 준 '과장 버전'을 함께 만든다.
"""
import os, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_datasets.font_square import font_transforms_split as FT
from custom_datasets.font_square.font_square_split import make_renderers, get_fonts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "img_pipeline"
OUT.mkdir(exist_ok=True)

FONT = ROOT.parent / "assets/fonts_korean_v2/train/NanumGothic.ttf"
BGS = ROOT.parent / "assets/backgrounds"

STYLE_TEXT = "스타일ABC"     # 왼쪽(style 참조)  (기본값, run() 인자로 덮어씀)
GEN_TEXT = "생성123"          # 오른쪽(gen 목표)

# ---------- 텐서 -> PIL 저장 헬퍼 ----------
def to_pil(img, normalized=False):
    t = img.detach().clone().float()
    if normalized:                       # Normalize(0.5,0.5) 되돌리기
        t = t * 0.5 + 0.5
    t = t.clamp(0, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def save(sample, step, normalized=False, cut_x=None):
    pil = to_pil(sample["img"], normalized=normalized)
    if cut_x is not None:                # 계산된 style/gen 경계선을 빨간 세로선으로
        d = ImageDraw.Draw(pil)
        d.line([(cut_x, 0), (cut_x, pil.height)], fill=(255, 0, 0), width=2)
    # 너무 작으면 3배 확대해서 보기 좋게
    if pil.height < 80:
        pil = pil.resize((pil.width * 3, pil.height * 3), Image.NEAREST)
    path = OUT / f"{step}.png"
    pil.save(path)
    _, h, w = sample["img"].shape
    print(f"saved {path.name:32s}  shape=(C,{h},{w})")
    return path

# ---------- 렌더러 준비 ----------
renderers = make_renderers([FONT], calib_threshold=0.8, verbose=False)
fonts = get_fonts([FONT])
print(f"renderers={len(renderers)}  charset_size={len(renderers[0].charset) if renderers[0].charset else 'None'}")

def run(seed, out_prefix, force=False, style_text=STYLE_TEXT, gen_text=GEN_TEXT,
        rot_deg=None, rot_p=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    global OUT
    OUT = ROOT / f"img_pipeline_{out_prefix}"
    OUT.mkdir(exist_ok=True)

    sample = {"style_text": style_text, "gen_text": gen_text, "font_id": 0}

    # 1) RenderImage : style/gen 각각 렌더 후 [style|20px|gen] 로 이어붙임
    render = FT.RenderImage(fonts, pad=20, renderers=renderers)
    sample = render(sample)
    print(f"\n[{out_prefix}] style_img_width={sample['style_img_width']} "
          f"gen_img_width={sample['gen_img_width']} total={sample['total_img_width']}")
    save(sample, "01_render_concat")

    # 2) RandomRotation
    _deg = rot_deg if rot_deg is not None else (8 if force else 3)
    _p = rot_p if rot_p is not None else (1.0 if force else 0.5)
    rot = FT.RandomRotation(_deg, fill=1, p=_p)
    sample = rot.forward(sample); save(sample, "02_rotation")

    # 3) RandomWarping (TPS)
    warp = FT.RandomWarping(std=0.08 if force else 0.05, grid_shape=(5, 2), p=1.0)
    sample = warp(sample); save(sample, "03_warping")

    # 4) GaussianBlur
    blur = FT.GaussianBlur(kernel_size=3, p=1.0)
    sample = blur(sample); save(sample, "04_blur")

    # 5) RandomBackground
    bg = FT.RandomBackground([p for p in BGS.rglob("*") if p.suffix in (".jpg", ".png", ".jpeg")], white_p=0.0)
    sample = bg(sample); save(sample, "05_background")   # 아직 img는 글자 흑백, bg_patch만 채워짐

    # 6) TailorTensor : 잉크 bbox 기준 좌우 여백 크롭  <-- 경계 어긋남의 핵심
    tail = FT.TailorTensor(pad=3)
    total_before = sample["total_img_width"]
    sample = tail(sample); save(sample, "06_tailor_crop")

    # 7) SplitAlphaChannel
    spl = FT.SplitAlphaChannel(); sample = spl(sample)

    # 8) GrayscaleDilation
    dil = FT.GrayscaleDilation(kernel_size=2, p=1.0); sample = dil(sample)

    # 9) ColorJitter
    cj = FT.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0, p=1.0)
    sample = cj.forward(sample)

    # 10) RandomInvert (p=0 으로 꺼둠: 원본 fork 기본 비활성)
    # 11) ImgResize(64)
    res = FT.ImgResize(64); sample = res(sample)

    # 12) MergeWithBackground : 글자*alpha 를 배경 위에 합성 -> 최종 컬러
    mrg = FT.MergeWithBackground(); sample = mrg(sample)
    save(sample, "07_merged_final")

    # ---- __getitem__ 의 경계 계산을 그대로 재현 ----
    W = sample["img"].shape[2]
    cut = sample["style_img_width"] * W // sample["total_img_width"]
    print(f"[{out_prefix}] 최종폭 W={W}, total_before_crop={total_before}, "
          f"계산된 cut_x={cut}  (style_img_width={sample['style_img_width']} 사용)")
    save(sample, "08_final_with_cut", cut_x=cut)

    # 13) 실제 분리 결과
    style_img = sample["img"][:, :, :cut]
    gen_img = sample["img"][:, :, cut:]
    to_pil(style_img).resize((style_img.shape[2]*3, style_img.shape[1]*3), Image.NEAREST).save(OUT/"09_style_part.png")
    to_pil(gen_img).resize((gen_img.shape[2]*3, gen_img.shape[1]*3), Image.NEAREST).save(OUT/"09_gen_part.png")
    print(f"[{out_prefix}] style_part width={style_img.shape[2]}  gen_part width={gen_img.shape[2]}")

LONG_STYLE = "오늘은 날씨가 맑고 바람이 시원하게 불어서 산책하기 좋은 날이었다"
LONG_GEN = "인공지능 모델이 손글씨 스타일을 학습하여 새로운 문장을 생성한다"

run(seed=7, out_prefix="normal", force=False)
run(seed=3, out_prefix="exaggerated", force=True)
# 긴 텍스트: 회전 약함(normal)과 회전 강함(exaggerated) 둘 다
run(seed=7, out_prefix="long_normal", force=False, style_text=LONG_STYLE, gen_text=LONG_GEN,
    rot_p=0.0)  # 회전 없이: 긴 텍스트여도 분리는 정상인지
run(seed=3, out_prefix="long_rot", force=True, style_text=LONG_STYLE, gen_text=LONG_GEN)  # 8°
run(seed=1, out_prefix="long_rot3", style_text=LONG_STYLE, gen_text=LONG_GEN,
    rot_deg=[3.0, 3.0], rot_p=1.0)  # 실제 학습 최대치인 정확히 3°로 고정
print("\nDONE")
