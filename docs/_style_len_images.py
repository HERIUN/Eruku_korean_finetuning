"""버그 시각화: '모델이 실제로 받는 style' 을 VAE 디코더로 복원해 이미지로 저장.
학습 체크포인트 불필요 — get_model_inputs 의 slice 로직만 재현해서 VAE decode 한다.

- style_00_original.png     : 렌더된 원본 style (전체)
- style_10_BEFORE_fix.png   : 수정 전 모델이 받는 style (latent 폭으로 clamp → 1/8)
- style_20_AFTER_fix.png    : 수정 후 모델이 받는 style (*8 → 전체)
gen 도 동일 원리이므로 style 로 대표.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL

from custom_datasets.font_square.render_font import Render

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "img_style_len"; OUT.mkdir(exist_ok=True)
FONT = ROOT.parent / "assets/fonts_korean_v2/train/NanumGothic.ttf"
STYLE_TEXT = "손글씨스타일참조ABC"      # 길게 → 잘림이 잘 보이도록
MAX_IMG_LEN = 8192

vae = AutoencoderKL.from_pretrained("blowing-up-groundhogs/emuru_vae").eval()

def normalize(t):                        # _img_encode 와 동일 (0.5,0.5)
    return (t - 0.5) / 0.5

def render_style(text):
    r = Render(FONT, font_size=64); r.calibrate(threshold=0.8)
    np_img, _ = r.render(text, action="top_left", pad=10)   # 흰(1) 배경, 검은(0) 글자
    t = torch.from_numpy(np_img).float().unsqueeze(0) / 255.0  # [1,H,W] in [0,1]
    # height 64 로 리사이즈 (학습 입력 규격)
    import torchvision.transforms.functional as F
    _, h, w = t.shape
    t = F.resize(t, [64, int(64 * w / h)], antialias=True)
    return t.repeat(3, 1, 1)             # VAE 는 3채널 입력 → 흑백 3ch 복제

def decode_to_pil(z):
    with torch.no_grad():
        img = vae.decode(z).sample       # 비대칭 VAE: 출력 1채널 [-1,1]
    img = ((img.clamp(-1, 1) + 1) / 2)[0]  # [C,64,W] in [0,1]
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def save(pil, name, scale=3):
    pil = pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)
    pil.save(OUT / name)
    print(f"saved {name:28s} width={pil.width//scale}px")

style = render_style(STYLE_TEXT)          # [3,64,W]
W = style.shape[-1]
with torch.no_grad():
    z = vae.encode(normalize(style.unsqueeze(0))).latent_dist.mode()   # [1,1,8,W/8]
latent_w = z.shape[-1]
print(f"style 원본 폭 W={W}px, latent 폭={latent_w}")

# 원본 style
save(decode_to_pil(z), "style_00_original.png")

# get_model_inputs 의 sl 계산 재현 (단일 샘플 = 추론 배치=1 = 최악의 경우)
sl_px = W
# --- BEFORE fix: min(sl, latent_w, max//2), slice [:sl//8]
sl_before = max(64, min(sl_px, latent_w, MAX_IMG_LEN // 2))
cols_before = max(1, sl_before // 8)
save(decode_to_pil(z[..., :cols_before]), "style_10_BEFORE_fix.png")

# --- AFTER fix: min(sl, latent_w*8, max//2), slice [:sl//8]
sl_after = max(64, min(sl_px, latent_w * 8, MAX_IMG_LEN // 2))
cols_after = max(1, sl_after // 8)
save(decode_to_pil(z[..., :cols_after]), "style_20_AFTER_fix.png")

print(f"\nBEFORE: sl={sl_before}px → latent {cols_before}/{latent_w} 열 사용 = {cols_before*8}px ({cols_before*8/W:.0%})")
print(f"AFTER : sl={sl_after}px → latent {cols_after}/{latent_w} 열 사용 = {cols_after*8}px ({cols_after*8/W:.0%})")
