"""long_normal(회전 없음) 케이스에서 계산된 cut 위치가 실제 잉크(글자)를 관통하는지 측정."""
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_datasets.upstream.font_square import font_transforms_split as FT
from custom_datasets.upstream.font_square.font_square_split import make_renderers, get_fonts

ROOT = Path(__file__).resolve().parent
FONT = ROOT.parent / "assets/fonts_korean_v2/train/NanumGothic.ttf"
BGS = ROOT.parent / "assets/backgrounds"
LONG_STYLE = "오늘은 날씨가 맑고 바람이 시원하게 불어서 산책하기 좋은 날이었다"
LONG_GEN = "인공지능 모델이 손글씨 스타일을 학습하여 새로운 문장을 생성한다"

renderers = make_renderers([FONT], calib_threshold=0.8, verbose=False)
fonts = get_fonts([FONT])

random.seed(7); np.random.seed(7); torch.manual_seed(7)
sample = {"style_text": LONG_STYLE, "gen_text": LONG_GEN, "font_id": 0}
sample = FT.RenderImage(fonts, pad=20, renderers=renderers)(sample)
# ▼ 데모 long_normal 과 완전히 동일한 체인 (회전만 OFF, 워핑은 p=1.0 로 켜짐)
sample = FT.RandomWarping(std=0.05, grid_shape=(5, 2), p=1.0)(sample)   # ← 이게 경계를 흔든다
sample = FT.GaussianBlur(kernel_size=3, p=1.0)(sample)
sample = FT.RandomBackground([p for p in BGS.rglob('*') if p.suffix in ('.jpg','.png','.jpeg')], white_p=0.0)(sample)
sample = FT.TailorTensor(pad=3)(sample)
sample = FT.SplitAlphaChannel()(sample)
sample = FT.GrayscaleDilation(kernel_size=2, p=1.0)(sample)
sample = FT.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0, p=1.0).forward(sample)
sample = FT.ImgResize(64)(sample)
sample = FT.MergeWithBackground()(sample)

img = sample["img"]                 # [3,64,W]
W = img.shape[2]
cut = sample["style_img_width"] * W // sample["total_img_width"]

# 컬럼별 잉크량(어두운 정도). 배경은 밝음(~1), 글자는 어두움(~0)
gray = img.mean(0)                  # [64,W]
ink = (gray < 0.5).float().sum(0).numpy()   # 각 열의 검은 픽셀 수

# 계산된 cut 주변에서 '진짜 흰 띠(gap)' = 잉크 0 인 열들의 최장 구간 찾기
def find_gap_near(x, radius=120):
    lo, hi = max(0, x-radius), min(W, x+radius)
    zeros = np.where(ink[lo:hi] == 0)[0]
    if len(zeros) == 0:
        return None
    # 최장 연속 구간
    runs, s = [], zeros[0]
    for a, b in zip(zeros, zeros[1:]):
        if b != a+1:
            runs.append((s, a)); s = b
    runs.append((s, zeros[-1]))
    best = max(runs, key=lambda r: r[1]-r[0])
    return best[0]+lo, best[1]+lo

gap = find_gap_near(cut)
print(f"최종폭 W={W}  계산된 cut={cut}")
print(f"cut 열의 잉크량 ink[cut]={ink[cut]:.0f}  (0이면 흰공간, >0이면 글자 관통)")
if gap:
    gc = (gap[0]+gap[1])//2
    print(f"실제 흰 띠(gap) 구간=({gap[0]}~{gap[1]}), 폭={gap[1]-gap[0]}px, 중앙={gc}")
    print(f"→ cut 은 실제 gap 중앙보다 {cut-gc:+d}px 어긋남 "
          f"(gap 밖이면 글자 침범)")
    print(f"→ cut 이 gap 안에 있나? {'예' if gap[0] <= cut <= gap[1] else '아니오 (글자 관통!)'}")

# 경계부 확대 저장: cut 은 빨강, 실제 gap 중앙은 초록
crop_lo, crop_hi = cut-90, cut+90
def to_pil(t):
    a = (t.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
    return Image.fromarray(a)
pil = to_pil(img[:, :, crop_lo:crop_hi])
pil = pil.resize((pil.width*5, pil.height*5), Image.NEAREST)
d = ImageDraw.Draw(pil)
d.line([((cut-crop_lo)*5,0),((cut-crop_lo)*5,pil.height)], fill=(255,0,0), width=2)
if gap:
    d.line([((gc-crop_lo)*5,0),((gc-crop_lo)*5,pil.height)], fill=(0,180,0), width=2)
pil.save(ROOT/"img_pipeline_long_normal/10_boundary_zoom.png")
print("saved 10_boundary_zoom.png (빨강=계산된 cut, 초록=실제 흰띠 중앙)")
