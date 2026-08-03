"""최소 전처리 vs 권장 전처리 — 공통 기준으로 정렬해 비교.

문제: 변형마다 해상도가 달라 '자기 입력 대비 MSE' 는 서로 비교 불가.
해결: 모든 복원을 **공통 캔버스(h=64 grayscale)** 로 리샘플해 같은 기준(원본의 h=64 판)과 비교.
      추가로 한글 HTR CER(리더가 내부에서 h=64 로 맞추므로 스케일 무관)도 같이 본다.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim

from _common import HTR_CKPT, TRAIN_FONTS, read_hw_labels, read_cer
from infer_show import render_in_font
from train_aux_htr import load_pretrained_htr

TEXTS = ["형형색색 꽃들이 활짝 피어난 봄날의 공원", "2024년 10월 15일 서울특별시 강남구 역삼동",
         "인공지능 모델의 한국어 생성 성능 평가", "다람쥐 헌 쳇바퀴에 타고파 오늘도 힘차게"]


def build_samples(n_font_files=3):
    fonts = sorted(TRAIN_FONTS.glob("*.ttf"))[:n_font_files]
    samples = [("font", Image.fromarray(render_in_font(f, t)), t) for t in TEXTS for f in fonts]
    for p, txt in read_hw_labels():
        samples.append(("scan", Image.open(p), txt))
    return samples


def canon(img, h=64):
    """공통 기준: 원본 → grayscale → h=64 (bicubic) → [0,1] tensor [1,1,64,W]"""
    g = img.convert("L")
    w = max(1, round(h * g.width / g.height))
    t = torch.from_numpy(np.array(g.resize((w, h), Image.BICUBIC), np.float32) / 255.0)
    return t[None, None]


def prep_min(img):
    """최소: grayscale → [0,1] → 3채널. 리사이즈/정규화/패딩 없음."""
    g = img.convert("L")
    t = torch.from_numpy(np.array(g, np.float32) / 255.0)[None, None]
    return t.repeat(1, 3, 1, 1), 0                       # (입력, 우측 패딩량)


def prep_resize_only(img, h=64):
    """리사이즈만: h=64, [0,1], 패딩/정규화 없음."""
    g = img.convert("L")
    w = max(1, round(h * g.width / g.height))
    t = torch.from_numpy(np.array(g.resize((w, h), Image.BICUBIC), np.float32) / 255.0)[None, None]
    return t.repeat(1, 3, 1, 1), 0


def prep_rec(img, h=64):
    """권장: h=64 bicubic + [-1,1] + 폭 8배수 흰색 패딩 + 3채널."""
    g = img.convert("L")
    w = max(1, round(h * g.width / g.height))
    t = torch.from_numpy(np.array(g.resize((w, h), Image.BICUBIC), np.float32) / 255.0)[None, None]
    pad = -w % 8
    x = F.pad(t * 2 - 1, (0, pad), value=1.0)
    return x.repeat(1, 3, 1, 1), pad


@torch.no_grad()
def run(name, prep, samples, vae, htr, dev):
    ms, ss, cers, floors, shapes = [], [], [], [], []
    for kind, img, text in samples:
        x, pad = prep(img)
        z = vae.encode(x.to(dev)).latent_dist.mode()
        out = vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu()
        rec01 = (out + 1) / 2                                   # 디코더는 항상 [-1,1] 도메인
        if pad:                                                  # 패딩분 잘라내 원래 폭으로
            rec01 = rec01[..., : rec01.shape[-1] - pad]
        shapes.append(tuple(x.shape[-2:]))

        # 공통 캔버스(h=64)로 정렬 후 기준과 비교
        ref = canon(img)
        r64 = F.interpolate(rec01, size=(64, max(1, round(64 * rec01.shape[-1] / rec01.shape[-2]))),
                            mode="bicubic", align_corners=False).clamp(0, 1)
        W = min(r64.shape[-1], ref.shape[-1])
        a = ref[0, 0, :, :W].numpy(); b = r64[0, 0, :, :W].numpy()
        ms.append(float(np.mean((a - b) ** 2))); ss.append(float(ssim(a, b, data_range=1.0)))

        rec_u8 = (rec01[0, 0].numpy() * 255).astype(np.uint8)
        ref_u8 = (ref[0, 0].numpy() * 255).astype(np.uint8)
        cers.append(read_cer(htr, rec_u8, text, dev))
        floors.append(read_cer(htr, ref_u8, text, dev))

    n_font = sum(1 for s in samples if s[0] == "font")
    def agg(v, sl):
        return float(np.mean(v[sl]))
    v_ms, v_ss, v_c, v_f = map(np.array, (ms, ss, cers, floors))
    print(f"\n{name}")
    print(f"  입력 크기 예: {shapes[0]} (font), {shapes[-1]} (scan)")
    for lbl, sl in (("font", slice(0, n_font)), ("scan", slice(n_font, None)), ("전체", slice(None))):
        print(f"  {lbl:5s} MSE={agg(v_ms, sl):.4f} SSIM={agg(v_ss, sl):.3f} "
              f"CER={agg(v_c, sl):.3f} (기준입력 {agg(v_f, sl):.3f}, 손상 {agg(v_c, sl)-agg(v_f, sl):+.3f})")
    return dict(name=name, mse=float(v_ms.mean()), ssim=float(v_ss.mean()),
                cer=float(v_c.mean()), floor=float(v_f.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vae", default="HERIUN/emuru_vae_korean", help="평가할 VAE(경로 또는 hub id)")
    ap.add_argument("--n-fonts", type=int, default=3, help="폰트 렌더에 쓸 폰트 수")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae).eval().to(dev)
    htr = load_pretrained_htr(HTR_CKPT).to(dev).eval()
    samples = build_samples(args.n_fonts)
    n_font = sum(1 for s in samples if s[0] == "font")
    print(f"samples: {n_font} font + {len(samples)-n_font} scan  "
          f"(원본 크기 예: {samples[-1][1].size})")

    rows = [run("① 최소 (원본 해상도, [0,1], 3ch)", prep_min, samples, vae, htr, dev),
            run("② 리사이즈만 (h64, [0,1])", prep_resize_only, samples, vae, htr, dev),
            run("③ 권장 (h64 + [-1,1] + 8배수 패딩)", prep_rec, samples, vae, htr, dev)]
    print(f"\n=== 요약(전체 {len(samples)}줄) ===")
    base = rows[0]
    for r in rows:
        d_mse = (r["mse"] / base["mse"] - 1) * 100
        print(f"  {r['name']:34s} MSE={r['mse']:.4f} ({d_mse:+.0f}%)  SSIM={r['ssim']:.3f}  "
              f"CER={r['cer']:.3f}  손상={r['cer']-r['floor']:+.3f}")


if __name__ == "__main__":
    main()
