"""최소 전처리 vs 권장 전처리 — 공통 기준으로 정렬해 비교.

문제: 변형마다 해상도가 달라 '자기 입력 대비 MSE' 는 서로 비교 불가.
해결: 모든 복원을 **공통 캔버스(h=64 grayscale)** 로 리샘플해 같은 기준(원본의 h=64 판)과 비교.
      추가로 한글 HTR CER(리더가 내부에서 h=64 로 맞추므로 스케일 무관)도 같이 본다.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim

REPO = Path("/data/work/dgkang/Eruku_korean_finetuning")
sys.path.insert(0, str(REPO))
from infer_show import render_in_font
from eval_htr_cer import to_htr_input, htr_read, norm
from train_aux_htr import load_pretrained_htr, cer as char_cer

dev = torch.device("cuda")
vae = AutoencoderKL.from_pretrained("HERIUN/emuru_vae_korean").eval().to(dev)
htr = load_pretrained_htr(REPO / "finetune_runs/aux_htr_ko/htr_s20000").to(dev).eval()

TEXTS = ["형형색색 꽃들이 활짝 피어난 봄날의 공원", "2024년 10월 15일 서울특별시 강남구 역삼동",
         "인공지능 모델의 한국어 생성 성능 평가", "다람쥐 헌 쳇바퀴에 타고파 오늘도 힘차게"]
fonts = sorted((REPO / "assets/fonts_korean_v2/train").glob("*.ttf"))[:3]
samples = [("font", Image.fromarray(render_in_font(f, t)), t) for t in TEXTS for f in fonts]
lab = {}
for ln in (REPO / "custom_datasets/hw_line_sample/label.txt").read_text(encoding="utf-8").splitlines():
    if "\t" in ln:
        k, v = ln.rstrip("\n").split("\t", 1)
        lab[k] = v
for p in sorted((REPO / "custom_datasets/hw_line_sample").glob("line_*.png")):
    samples.append(("scan", Image.open(p), lab[p.name]))
print(f"samples: {sum(1 for s in samples if s[0]=='font')} font + "
      f"{sum(1 for s in samples if s[0]=='scan')} scan  (원본 크기 예: {samples[-1][1].size})")


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
def run(name, prep):
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
        cers.append(char_cer([norm(htr_read(htr, to_htr_input(rec_u8, dev), dev)[0])], [norm(text)]))
        floors.append(char_cer([norm(htr_read(htr, to_htr_input(ref_u8, dev), dev)[0])], [norm(text)]))

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


rows = [run("① 최소 (원본 해상도, [0,1], 3ch)", prep_min),
        run("② 리사이즈만 (h64, [0,1])", prep_resize_only),
        run("③ 권장 (h64 + [-1,1] + 8배수 패딩)", prep_rec)]
print("\n=== 요약(전체 32줄) ===")
base = rows[0]
for r in rows:
    d_mse = (r["mse"] / base["mse"] - 1) * 100
    print(f"  {r['name']:34s} MSE={r['mse']:.4f} ({d_mse:+.0f}%)  SSIM={r['ssim']:.3f}  "
          f"CER={r['cer']:.3f}  손상={r['cer']-r['floor']:+.3f}")
