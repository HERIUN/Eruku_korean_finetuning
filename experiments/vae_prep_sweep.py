"""VAE 입력 전처리 스윕 — 최소 조건 + 성능 최적 전처리 찾기.

각 변형마다:
  self MSE/SSIM  = 복원 vs 그 변형의 입력 (latent 가 입력을 얼마나 충실히 담는가)
  CER(recon)     = 한글 HTR 리더가 복원 이미지를 읽은 CER
  CER(input)     = 같은 리더가 입력 이미지를 읽은 CER (= 바닥)
  damage         = CER(recon) - CER(input)  ← VAE 통과로 잃은 가독성만 분리
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim

from common import HTR_CKPT, TRAIN_FONTS, read_hw_labels, read_cer
from infer.show import render_in_font
from train.aux_htr import load_pretrained_htr

# ── 샘플: 깨끗한 폰트 렌더 + 실제 스캔 ────────────────────────────────────────
FONTS = ["NanumGothic", "GowunDodum", "Pretendard-Regular"]
TEXTS = ["형형색색 꽃들이 활짝 피어난 봄날의 공원",
         "2024년 10월 15일 서울특별시 강남구 역삼동",
         "인공지능 모델의 한국어 생성 성능 평가"]


def build_samples(n_scan):
    samples = []
    for t in TEXTS:
        for f in FONTS[:2]:
            fp = TRAIN_FONTS / f"{f}.ttf"
            if fp.exists():
                samples.append(("font", Image.fromarray(render_in_font(fp, t)), t))
    for p, txt in read_hw_labels()[:n_scan]:
        samples.append(("scan", Image.open(p), txt))
    return samples


RESAMPLE = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}


def prep(img, height=64, interp="bilinear", contrast=None, pad8=True, invert=False,
         torch_interp=False, antialias=True):
    """PIL → (x[1,3,h,W] VAE 입력, in01[1,1,h,W] 같은 스케일의 입력 기록)"""
    g = img.convert("L")
    if invert:
        g = ImageOps.invert(g)
    w = max(1, round(height * g.width / g.height))
    if torch_interp:
        t = torch.from_numpy(np.array(g, np.float32) / 255.0)[None, None]
        t = F.interpolate(t, size=(height, w), mode=interp,
                          align_corners=False if interp != "nearest" else None,
                          antialias=antialias if interp in ("bilinear", "bicubic") else False)
        t = t.clamp(0, 1)
    else:
        g = g.resize((w, height), RESAMPLE[interp])
        t = torch.from_numpy(np.array(g, np.float32) / 255.0)[None, None]

    if contrast == "stretch":                       # min-max 대비 스트레치
        lo, hi = float(t.min()), float(t.max())
        if hi - lo > 1e-3:
            t = (t - lo) / (hi - lo)
    elif contrast == "autocontrast":                # PIL autocontrast (1% cutoff)
        a = (t[0, 0].numpy() * 255).astype(np.uint8)
        a = np.array(ImageOps.autocontrast(Image.fromarray(a), cutoff=1))
        t = torch.from_numpy(a.astype(np.float32) / 255.0)[None, None]
    elif contrast == "binarize":                    # Otsu 이진화
        a = (t[0, 0].numpy() * 255).astype(np.uint8)
        hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
        tot = hist.sum(); w0 = np.cumsum(hist); m = np.cumsum(hist * np.arange(256))
        mt = m[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (mt * w0 / tot - m) ** 2 / (w0 * (tot - w0))
        thr = int(np.nanargmax(var))
        t = torch.from_numpy((a > thr).astype(np.float32))[None, None]

    x01 = t
    x = x01 * 2 - 1
    if pad8:
        x = F.pad(x, (0, -w % 8), value=1.0)
        x01 = F.pad(x01, (0, -w % 8), value=1.0)
    return x.repeat(1, 3, 1, 1), x01


@torch.no_grad()
def evaluate(name, samples, vae, htr, dev, **kw):
    ms, ss, cr, ci = [], [], [], []
    err = None
    for kind, img, text in samples:
        try:
            x, in01 = prep(img, **kw)
            z = vae.encode(x.to(dev)).latent_dist.mode()
            out = vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu()
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:70]}"
            break
        rec01 = (out + 1) / 2
        W = min(rec01.shape[-1], in01.shape[-1]); H = min(rec01.shape[-2], in01.shape[-2])
        a = in01[0, 0, :H, :W].numpy(); b = rec01[0, 0, :H, :W].numpy()
        ms.append(float(np.mean((a - b) ** 2))); ss.append(float(ssim(a, b, data_range=1.0)))
        rec_u8 = (b * 255).astype(np.uint8); in_u8 = (a * 255).astype(np.uint8)
        if kw.get("invert"):                       # 리더는 어두운 잉크/밝은 배경을 기대
            rec_u8 = 255 - rec_u8; in_u8 = 255 - in_u8
        cr.append(read_cer(htr, rec_u8, text, dev))
        ci.append(read_cer(htr, in_u8, text, dev))
    if err:
        print(f"{name:38s} FAIL  {err}")
        return None
    r = dict(name=name, mse=np.mean(ms), ssim=np.mean(ss), cer=np.mean(cr), floor=np.mean(ci))
    r["damage"] = r["cer"] - r["floor"]
    print(f"{name:38s} MSE={r['mse']:.4f} SSIM={r['ssim']:.3f} "
          f"CER={r['cer']:.3f} (입력바닥 {r['floor']:.3f}, 손상 {r['damage']:+.3f})")
    return r


VARIANTS = [
    ("기준: PIL bilinear / pad8",            dict()),
    ("PIL bicubic",                          dict(interp="bicubic")),
    ("PIL lanczos",                          dict(interp="lanczos")),
    ("torch bilinear+antialias",             dict(torch_interp=True, interp="bilinear")),
    ("torch bilinear (antialias 없음)",        dict(torch_interp=True, interp="bilinear", antialias=False)),
    ("torch bicubic+antialias",              dict(torch_interp=True, interp="bicubic")),
    ("+ 대비 스트레치",                        dict(contrast="stretch")),
    ("+ autocontrast",                       dict(contrast="autocontrast")),
    ("+ Otsu 이진화",                         dict(contrast="binarize")),
    ("pad8 없음",                             dict(pad8=False)),
    ("height 96",                            dict(height=96)),
    ("반전 입력(흰글씨/검은배경)",                dict(invert=True)),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae", default="HERIUN/emuru_vae_korean", help="평가할 VAE(경로 또는 hub id)")
    ap.add_argument("--n-scan", type=int, default=6, help="실제 손글씨 스캔 표본 수")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae).eval().to(dev)
    htr = load_pretrained_htr(HTR_CKPT).to(dev).eval()
    samples = build_samples(args.n_scan)
    print(f"samples: {sum(1 for s in samples if s[0]=='font')} font + "
          f"{sum(1 for s in samples if s[0]=='scan')} scan")

    rows = [r for r in (evaluate(n, samples, vae, htr, dev, **kw) for n, kw in VARIANTS) if r]
    print("\n=== 손상(damage) 낮은 순 ===")
    for r in sorted(rows, key=lambda r: (round(r["damage"], 3), r["mse"])):
        print(f"  {r['name']:38s} damage={r['damage']:+.3f}  CER={r['cer']:.3f}  "
              f"MSE={r['mse']:.4f}  SSIM={r['ssim']:.3f}")


if __name__ == "__main__":
    main()
