"""[-1,1] vs [0,1] 정규화: (A) VAE 복원, (B) **Eruku 생성** 까지 검증.

(B) 가 핵심 — 학습은 [-1,1] 로 했으므로 style 조건 latent 가 미세하게 달라지면
T5 생성 품질이 떨어질 수 있다. 복원만 보고 '차이 없다'고 말할 근거는 안 된다.

실측 결론(2026-07-30): 둘 다 차이 없음. 복원 n=32 에서 MSE 0.0058([-1,1]) vs 0.0048([0,1]),
생성 n=24 에서 CER 0.046 vs 0.045(24개 중 21개가 동일). 첫 conv 직후 GroupNorm 이 global
shift/scale 을 흡수하기 때문. 단 0~255 raw 처럼 스케일이 크게 벗어나면 latent 가 깨진다.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from _common import REPO, HTR_CKPT, TRAIN_FONTS, HW_DIR, read_cer
from infer_show import load_model, render_in_font, style_tensor, gen_from_style
from eval_echo_metrics import build_texts, font_can_render
from train_aux_htr import load_pretrained_htr

DEF_CKPT = REPO / "finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth"
LABELS = ("[-1,1] (학습 규약)", "[0,1]")

# ── (A) 복원용 고정 probe ──────────────────────────────────────────────────────
TEXTS_R = ["형형색색 꽃들이 활짝 피어난 봄날의 공원", "2024년 10월 15일 서울특별시 강남구",
           "인공지능 모델의 한국어 생성 성능 평가", "밝은 햇살 붉은 노을 넓은 들판"]


def build_recon_imgs(n_fonts=3):
    """폰트 렌더 + 실제 손글씨 스캔 전부."""
    fonts = sorted(TRAIN_FONTS.glob("*.ttf"))[:n_fonts]
    imgs = [Image.fromarray(render_in_font(f, t)) for t in TEXTS_R for f in fonts]
    imgs += [Image.open(p) for p in sorted(HW_DIR.glob("line_*.png"))]
    return imgs


@torch.no_grad()
def recon(vae, dev, img, norm11):
    """한 장 roundtrip → (MSE, SSIM, latent). norm11=False 면 [0,1] 을 그대로 넣는다."""
    g = img.convert("L")
    w = max(1, round(64 * g.width / g.height))
    t = torch.from_numpy(np.array(g.resize((w, 64), Image.BICUBIC), np.float32) / 255.)[None, None]
    x01 = F.pad(t, (0, -w % 8), value=1.0)
    x = (x01 * 2 - 1) if norm11 else x01
    z = vae.encode(x.repeat(1, 3, 1, 1).to(dev)).latent_dist.mode()
    rec = ((vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu() + 1) / 2)
    a, b = x01[0, 0].numpy(), rec[0, 0].numpy()
    return float(np.mean((a - b) ** 2)), float(ssim(a, b, data_range=1.0)), z.cpu()


def run_recon(vae, dev, imgs):
    """(A) 복원 비교 + 같은 입력에서 두 스케일의 latent 차이."""
    for label, n11 in zip(LABELS, (True, False)):
        ms, ss, _ = zip(*[recon(vae, dev, im, n11) for im in imgs])
        print(f"(A) 복원 {label:16s} MSE={np.mean(ms):.4f} SSIM={np.mean(ss):.3f}  (n={len(imgs)})")
    z11 = recon(vae, dev, imgs[0], True)[2]
    z01 = recon(vae, dev, imgs[0], False)[2]
    cos = float((z11.flatten() @ z01.flatten()) / (z11.norm() * z01.norm()))
    print(f"    latent 비교: cos={cos:.5f} meandiff={float((z11 - z01).abs().mean()):.4f} "
          f"maxdiff={float((z11 - z01).abs().max()):.4f}")


def run_generation(model, htr, dev, n, cfg, max_new):
    """(B) 같은 style/텍스트로 생성하되 조건 latent 의 입력 스케일만 다르게."""
    rng = random.Random(0)
    texts = build_texts(n, rng, english=False, coherent=True, max_words=15)
    fonts = sorted(TRAIN_FONTS.glob("*.ttf"))
    res = {lbl: [] for lbl in LABELS}
    floor = []
    for text in texts:
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text)), None)
        if fp is None:
            continue
        gt = render_in_font(fp, text)
        st11 = style_tensor(gt).unsqueeze(0).to(dev)      # [-1,1]
        st01 = (st11 + 1) / 2                              # [0,1]
        floor.append(read_cer(htr, gt, text, dev))
        for label, st in zip(LABELS, (st11, st01)):
            try:
                gen = gen_from_style(model, st, text, text, cfg, max_new, dev)
            except Exception as e:
                print("  [skip]", e)
                continue
            res[label].append(read_cer(htr, gen, text, dev))
        done = len(res[LABELS[1]])
        if done and done % 6 == 0:
            print(f"  ... {done}/{n}  " + "  ".join(f"{l}={np.mean(res[l]):.3f}" for l in LABELS))

    print(f"\n(B) Eruku 생성 CER (n={len(res[LABELS[1]])}, cfg={cfg}, GT 바닥 {np.mean(floor):.3f})")
    for label in LABELS:
        print(f"    {label:16s} CER={np.mean(res[label]):.3f}")
    m = len(res[LABELS[1]])
    d = np.array(res[LABELS[1]]) - np.array(res[LABELS[0]][:m])
    print(f"    차이([0,1] - [-1,1]) 평균={d.mean():+.3f}  악화 {int((d > 0).sum())}/{len(d)}  "
          f"동일 {int((d == 0).sum())}  개선 {int((d < 0).sum())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(DEF_CKPT), help="Eruku ckpt (내장 VAE 를 그대로 사용)")
    ap.add_argument("--n", type=int, default=24, help="(B) 생성 비교 표본 수")
    ap.add_argument("--n-fonts", type=int, default=3, help="(A) 복원 probe 폰트 수")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--skip-generation", action="store_true", help="(A) 복원만 재고 종료")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, step = load_model(args.ckpt, dev)          # ckpt 내장 한글 VAE 사용
    htr = load_pretrained_htr(HTR_CKPT).to(dev).eval()
    print(f"eruku step={step}")

    run_recon(model.vae, dev, build_recon_imgs(args.n_fonts))
    if not args.skip_generation:
        run_generation(model, htr, dev, args.n, args.cfg, args.max_new_tokens)


if __name__ == "__main__":
    main()
