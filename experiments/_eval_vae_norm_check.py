"""[-1,1] vs [0,1] 정규화: (A) VAE 복원, (B) **Eruku 생성** 까지 검증.

(B) 가 핵심 — 학습은 [-1,1] 로 했으므로 style 조건 latent 가 미세하게 달라지면
T5 생성 품질이 떨어질 수 있다. 복원만 보고 '차이 없다'고 말할 근거는 안 된다.
"""
import sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim

REPO = Path("/data/work/dgkang/Eruku_korean_finetuning")
sys.path.insert(0, str(REPO))
from infer_show import load_model, render_in_font, gen_arr
from eval_echo_metrics import build_texts, style_tensor, font_can_render
from eval_htr_cer import to_htr_input, htr_read, norm
from train_aux_htr import load_pretrained_htr, cer as char_cer

dev = torch.device("cuda")
CKPT = REPO / "finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth"
model, step = load_model(str(CKPT), dev)          # ckpt 내장 한글 VAE 사용
htr = load_pretrained_htr(REPO / "finetune_runs/aux_htr_ko/htr_s20000").to(dev).eval()
vae = model.vae
print(f"eruku step={step}")

# ── (A) 복원: n=32 ────────────────────────────────────────────────────────────
TEXTS_R = ["형형색색 꽃들이 활짝 피어난 봄날의 공원", "2024년 10월 15일 서울특별시 강남구",
           "인공지능 모델의 한국어 생성 성능 평가", "밝은 햇살 붉은 노을 넓은 들판"]
fonts_r = sorted((REPO / "assets/fonts_korean_v2/train").glob("*.ttf"))[:3]
imgs = [Image.fromarray(render_in_font(f, t)) for t in TEXTS_R for f in fonts_r]
imgs += [Image.open(p) for p in sorted((REPO / "custom_datasets/hw_line_sample").glob("line_*.png"))]

@torch.no_grad()
def recon(img, norm11):
    g = img.convert("L"); w = max(1, round(64 * g.width / g.height))
    t = torch.from_numpy(np.array(g.resize((w, 64), Image.BICUBIC), np.float32) / 255.)[None, None]
    x01 = F.pad(t, (0, -w % 8), value=1.0)
    x = (x01 * 2 - 1) if norm11 else x01
    z = vae.encode(x.repeat(1, 3, 1, 1).to(dev)).latent_dist.mode()
    rec = ((vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu() + 1) / 2)
    a, b = x01[0, 0].numpy(), rec[0, 0].numpy()
    return float(np.mean((a - b) ** 2)), float(ssim(a, b, data_range=1.0)), z.cpu()

rows = {}
for label, n11 in (("[-1,1] (학습 규약)", True), ("[0,1]", False)):
    ms, ss = [], []
    for im in imgs:
        m, s, _ = recon(im, n11)
        ms.append(m); ss.append(s)
    rows[label] = (np.mean(ms), np.mean(ss))
    print(f"(A) 복원 {label:16s} MSE={np.mean(ms):.4f} SSIM={np.mean(ss):.3f}  (n={len(imgs)})")
_, _, z11 = recon(imgs[0], True)
_, _, z01 = recon(imgs[0], False)
cos = float((z11.flatten() @ z01.flatten()) / (z11.norm() * z01.norm()))
print(f"    latent 비교: cos={cos:.5f} meandiff={float((z11-z01).abs().mean()):.4f} "
      f"maxdiff={float((z11-z01).abs().max()):.4f}")

# ── (B) Eruku 생성: 같은 style/텍스트, 조건 latent 만 스케일 다르게 ──────────────
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
rng = random.Random(0)
texts = build_texts(N, rng, english=False, coherent=True, max_words=15)
fonts = sorted((REPO / "assets/fonts_korean_v2/train").glob("*.ttf"))
res = {"[-1,1] (학습 규약)": [], "[0,1]": []}
floor = []
for text in texts:
    rng.shuffle(fonts)
    fp = next((f for f in fonts if font_can_render(f, text)), None)
    if fp is None:
        continue
    gt = render_in_font(fp, text)
    st11 = style_tensor(gt).unsqueeze(0).to(dev)          # [-1,1]
    st01 = (st11 + 1) / 2                                  # [0,1]
    floor.append(char_cer([norm(htr_read(htr, to_htr_input(gt, dev), dev)[0])], [norm(text)]))
    for label, st in (("[-1,1] (학습 규약)", st11), ("[0,1]", st01)):
        mi = model.get_model_inputs([st[0]], None, style_len=st.shape[-1], gen_len=None, max_img_len=2048)
        dec = mi["decoder_inputs_embeds"].to(dev); spx = dec.shape[1] * 8
        try:
            gen = gen_arr(model, dec, text, text, 1.0, 480, spx, dev)
        except Exception as e:
            print("  [skip]", e); continue
        res[label].append(char_cer([norm(htr_read(htr, to_htr_input(gen, dev), dev)[0])], [norm(text)]))
    if len(res["[0,1]"]) % 6 == 0 and res["[0,1]"]:
        print(f"  ... {len(res['[0,1]'])}/{N}  "
              f"[-1,1]={np.mean(res['[-1,1] (학습 규약)']):.3f}  [0,1]={np.mean(res['[0,1]']):.3f}")

print(f"\n(B) Eruku 생성 CER (n={len(res['[0,1]'])}, cfg=1.0, GT 바닥 {np.mean(floor):.3f})")
for label, v in res.items():
    print(f"    {label:16s} CER={np.mean(v):.3f}")
d = np.array(res["[0,1]"]) - np.array(res["[-1,1] (학습 규약)"][:len(res["[0,1]"])])
print(f"    차이([0,1] - [-1,1]) 평균={d.mean():+.3f}  악화한 샘플 {int((d>0).sum())}/{len(d)}  "
      f"동일 {int((d==0).sum())}  개선 {int((d<0).sum())}")
