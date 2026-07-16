"""VAE 한글 수용성 근거 이미지: 같은 폰트로 한글/영어를 VAE roundtrip(encode→decode) 시켜
원본 대비 손실을 나란히 비교. 한글(밀집 2D 음절)이 영어(선형 단순 글자)보다 더 뭉개짐을
정량(MSE↑/SSIM↓)·시각으로 보인다. 모델은 이 VAE latent 만 조건/타깃으로 쓰므로,
VAE 의 한글 재구성 한계 = 한글 학습의 하드 상한(= 어느 지점 이후 한글이 더 안 좋아지는 이유).

출력: docs/vae_korean_vs_english.png  (README 근거용)
"""
import sys
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim

HERE = Path(__file__).resolve().parent.parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("blowing-up-groundhogs/emuru_vae").to(dev).eval()
FONT = ImageFont.truetype(str(HERE / "assets/fonts_korean_v2/train/NanumGothic.ttf"), 42)
LAB = ImageFont.truetype(str(HERE / "assets/fonts_label/NanumGothic-Regular.ttf"), 16)

KO = ["다람쥐 헌 쳇바퀴에 타고파", "인공지능 모델 성능 평가", "형형색색 꽃들이 활짝"]
EN = ["The quick brown fox jumps", "evaluate model performance", "colorful flowers bloom"]
H = 64


def render(text, h=H):
    b = FONT.getbbox(text); w = b[2] - b[0]
    im = Image.new("L", (w + 24, 72), 255)
    ImageDraw.Draw(im).text((12 - b[0], 36 - (b[1] + b[3]) // 2), text, font=FONT, fill=0)
    a = np.array(im)
    ys, xs = np.where(a < 250)                       # 잉크 crop
    a = a[max(0, ys.min() - 6):ys.max() + 6, max(0, xs.min() - 6):xs.max() + 6]
    a = cv2.resize(a, (max(1, int(a.shape[1] * h / a.shape[0])), h))
    return a                                          # [h,W] uint8 grayscale


@torch.no_grad()
def roundtrip(gray):                                  # gray [h,W] uint8 → VAE 왕복 recon [h,W] float[0,1]
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)[None].repeat(3, 1, 1)  # [3,h,W] [0,1]
    x = (t * 2 - 1).unsqueeze(0).to(dev)                                          # [-1,1]
    z = vae.encode(x).latent_dist.mode()
    rec = vae.decode(z).sample.clamp(-1, 1)[0].mean(0).cpu().numpy()              # [h,W] [-1,1]
    return (rec + 1) / 2                                                          # [0,1]


rows = []
sep = np.full((3, 10), 255, np.uint8)
for lang, texts in (("KO", KO), ("EN", EN)):
    for text in texts:
        o = render(text)                              # [h,W] uint8
        r = roundtrip(o)                              # [h,W] float[0,1]
        w = min(o.shape[1], r.shape[1]); o1 = o[:, :w].astype(np.float32) / 255; r1 = r[:, :w]
        mse = float(np.mean((o1 - r1) ** 2))
        ss = float(ssim(o1, r1, data_range=1.0))
        rec_u = (np.clip(r, 0, 1) * 255).astype(np.uint8)
        # 블록: 라벨 + 원본 / VAE복원
        W = max(o.shape[1], rec_u.shape[1]) + 8
        head = Image.new("L", (W + 320, 22), 240)
        ImageDraw.Draw(head).text((6, 3), f"[{lang}] {text}   MSE={mse:.4f}  SSIM={ss:.3f}", font=LAB, fill=0)
        def strip(a, name):
            row = np.full((H, W + 320), 255, np.uint8)
            lab = Image.new("L", (86, H), 232); ImageDraw.Draw(lab).text((6, H // 2 - 9), name, font=LAB, fill=0)
            row[:, :86] = np.array(lab); row[:, 90:90 + a.shape[1]] = a
            return row
        block = np.vstack([np.array(head), strip(o, "원본"),
                           np.full((2, W + 320), 200, np.uint8), strip(rec_u, "VAE복원"),
                           np.full((6, W + 320), 90, np.uint8)])
        rows.append((block, mse, ss, lang))

# 폭 통일 + 세로 결합
Wm = max(b.shape[1] for b, *_ in rows)
imgs = [np.hstack([b, np.full((b.shape[0], Wm - b.shape[1]), 255, np.uint8)]) if b.shape[1] < Wm else b for b, *_ in rows]
out = HERE / "docs/vae_korean_vs_english.png"
img = np.vstack(imgs)
img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)  # 2x 확대
Image.fromarray(img).save(out)

ko = [(m, s) for _, m, s, l in rows if l == "KO"]; en = [(m, s) for _, m, s, l in rows if l == "EN"]
print(f"KO  평균 MSE={np.mean([m for m,_ in ko]):.4f}  SSIM={np.mean([s for _,s in ko]):.3f}")
print(f"EN  평균 MSE={np.mean([m for m,_ in en]):.4f}  SSIM={np.mean([s for _,s in en]):.3f}")
print(f"saved {out}  {img.shape[1]}x{img.shape[0]}")
