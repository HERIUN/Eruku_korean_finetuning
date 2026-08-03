"""기존 VAE vs 학습(한글적응) VAE 의 **한글 복원 능력** 비교 이미지.

순수 VAE roundtrip(encode.mode → decode) 만 본다(Eruku 생성 무관). 한 텍스트마다
[원본 | 기존VAE 복원 | 학습VAE 복원] 을 세로로 쌓고 각 복원의 MSE↓/SSIM↑ 를 표시.
한글은 밀집 2D 음절이라 기존 VAE 가 뭉개던 게(특히 작고 빽빽한 글자) 학습 VAE 에서 얼마나
선명해지는지 눈+숫자로 확인.

예:
  CUDA_VISIBLE_DEVICES=2 python _eval_vae_recon.py \
    --vae-new finetune_runs/vae_korean_dec/vae_s15000 \
    --out finetune_runs/korean_en2ko_fixed/_eval/recon_ko_old_vs_new.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL
from skimage.metrics import structural_similarity as ssim

HERE = Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(HERE))
from infer_show import render_in_font

TRAIN = HERE / "assets/fonts_korean_v2/train"
UNSEEN = HERE / "assets/fonts_korean_v2/test"
LABEL_FONT = HERE / "assets/fonts_label/NanumGothic-Regular.ttf"

# (폰트, 텍스트): 단문→장문 + 밀집/난해 음절(기존 VAE 가 잘 뭉개던 류) 포함. 마지막 2개는 held-out 폰트로 일반화 확인.
SAMPLES = [
    (TRAIN / "NanumGothic.ttf", "한국어 평가"),
    (TRAIN / "NanumGothic.ttf", "인공지능 모델 성능 평가"),
    (TRAIN / "NanumGothic.ttf", "형형색색 꽃들이 활짝 피어난 봄날의 공원"),
    (TRAIN / "NanumGothic.ttf", "료냉귀 내륙으로도 6,506원 봉떤텝쌉 술함을"),
    (TRAIN / "NanumGothic.ttf", "다람쥐 헌 쳇바퀴에 타고파 오늘도 힘차게 달린다"),
    (TRAIN / "NanumGothic.ttf", "2024년 10월 15일 서울특별시 강남구 역삼동"),
    (UNSEEN / "DoHyeon.ttf", "홀드아웃 폰트 일반화 확인 결과"),
    (UNSEEN / "GowunBatang.ttf", "학습에 안 쓴 폰트로도 복원 개선"),
]
# 영어 회귀 확인용(한글 적응이 영어 복원을 해쳤는지). 같은 폰트로 라틴 렌더.
EN_SAMPLES = [
    (TRAIN / "NanumGothic.ttf", "Hello World"),
    (TRAIN / "NanumGothic.ttf", "evaluate model performance"),
    (TRAIN / "NanumGothic.ttf", "The quick brown fox jumps over the lazy dog"),
    (TRAIN / "NanumGothic.ttf", "Autoregressive text image generation 2024"),
    (UNSEEN / "DoHyeon.ttf", "held out font generalization check"),
    (UNSEEN / "GowunBatang.ttf", "colorful flowers bloom in spring"),
]
# 어려운(극단 스타일) 폰트: 초굵은 display + 붓글씨 + 장식 → 1채널 latent 가 가장 못 담는 케이스
HARD_SAMPLES = [
    (UNSEEN / "BlackHanSans.ttf", "블랙한산스 초굵은 디스플레이"),
    (UNSEEN / "BagelFatOne.ttf", "베이글 통통한 라운드 폰트"),
    (UNSEEN / "NanumBrushScript.ttf", "나눔 붓글씨 흘림체 한국어"),
    (UNSEEN / "EastSeaDokdo.ttf", "동해 독도 손글씨 스크롤"),
    (TRAIN / "GasoekOne.ttf", "가석원 극단 헤비 폰트"),
    (TRAIN / "Ghanachocolate.ttf", "가나초콜릿 장식 폰트 복원"),
]
H = 64


def load_vae(src, dev):
    return AutoencoderKL.from_pretrained(src).to(dev).eval()


def render64(font, text):
    """text → clean bw [1,64,w] [-1,1] (bg=+1, ink=-1)."""
    arr = render_in_font(font, text)                       # [H,W] uint8 ink~0 bg~255
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)[None, None]
    w64 = max(1, int(round(H * arr.shape[1] / arr.shape[0])))
    t = F.interpolate(t, size=(H, w64), mode="bilinear", align_corners=False).clamp(0, 1)
    x = t * 2 - 1                                          # [1,1,64,w] [-1,1]
    w8 = (w64 + 7) // 8 * 8                                # VAE 는 폭이 8배수여야 정확 roundtrip → 흰색(+1) 우측패딩
    if w8 != w64:
        x = F.pad(x, (0, w8 - w64), value=1.0)
    return x


@torch.no_grad()
def roundtrip(vae, x, dev):
    z = vae.encode(x.repeat(1, 3, 1, 1).to(dev)).latent_dist.mode()
    return vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu()  # [1,1,64,w]


def metrics(orig, rec):
    a = ((orig[0, 0] + 1) / 2).numpy(); b = ((rec[0, 0] + 1) / 2).numpy()
    return float(np.mean((a - b) ** 2)), float(ssim(a, b, data_range=1.0))


def to_u8(t):
    return ((t[0, 0] + 1) / 2 * 255).clamp(0, 255).byte().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help='"라벨=경로" 목록. 각 텍스트마다 원본 아래로 이 순서대로 복원 stack. '
                         '예: 원본=blowing-up-groundhogs/emuru_vae decoder=.../vae_dec/vae_s15000 full=.../vae_full/vae_s15000')
    ap.add_argument("--out", default=str(HERE / "finetune_runs/korean_en2ko_fixed/_eval/recon_ko.png"))
    ap.add_argument("--lang", choices=["ko", "en", "hard"], default="ko", help="ko=한글 / en=영어 / hard=극단폰트")
    ap.add_argument("--label-lang", choices=["ko", "en"], default="ko",
                    help="figure 안 고정 라벨/제목 언어. en = 공개(HF 모델카드) 용")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    samples = {"en": EN_SAMPLES, "hard": HARD_SAMPLES}.get(args.lang, SAMPLES)

    specs = [m.split("=", 1) for m in args.models]        # [(label, path), ...]
    vaes = [(lbl, load_vae(path, dev)) for lbl, path in specs]
    lab = ImageFont.truetype(str(LABEL_FONT), 15) if LABEL_FONT.exists() else ImageFont.load_default()
    hdrf = ImageFont.truetype(str(LABEL_FONT), 17) if LABEL_FONT.exists() else ImageFont.load_default()

    LW = 110                                               # 좌측 라벨 폭
    blocks = []
    agg = {lbl: {"m": [], "s": []} for lbl, _ in vaes}     # 모델별 MSE/SSIM 누적
    for font, text in samples:
        if not Path(font).exists():
            print(f"[skip] font 없음 {font}"); continue
        x = render64(font, text)
        w = x.shape[-1]
        def strip(t_u8, name, extra=""):
            row = np.full((H, LW + w), 255, np.uint8)
            row[:, LW:LW + t_u8.shape[1]] = t_u8
            im = Image.fromarray(row)
            d = ImageDraw.Draw(im); d.text((6, 4), name, font=lab, fill=0)
            if extra: d.text((6, H - 22), extra, font=lab, fill=0)
            return np.array(im)
        head = Image.new("L", (LW + w, 24), 235)
        ImageDraw.Draw(head).text((6, 3), f"[{Path(font).stem}] {text}", font=hdrf, fill=0)
        parts = [np.array(head), strip(to_u8(x), "원본" if args.label_lang == "ko" else "source")]
        for lbl, vae in vaes:
            r = roundtrip(vae, x, dev)
            m, s = metrics(x, r)
            agg[lbl]["m"].append(m); agg[lbl]["s"].append(s)
            parts += [np.full((2, LW + w), 210, np.uint8), strip(to_u8(r), lbl, f"MSE {m:.4f} SSIM {s:.3f}")]
        parts.append(np.full((8, LW + w), 90, np.uint8))
        blocks.append(np.vstack(parts))

    Wm = max(b.shape[1] for b in blocks)
    blocks = [np.hstack([b, np.full((b.shape[0], Wm - b.shape[1]), 255, np.uint8)]) if b.shape[1] < Wm else b
              for b in blocks]
    # 요약 헤더: 모델별 평균 MSE/SSIM + 첫 모델(보통 원본) 대비 증감
    if args.label_lang == "ko":
        _langname = {"en": "영어", "hard": "극단폰트"}.get(args.lang, "한글")
        _title = f"{_langname} VAE 복원 비교 (roundtrip, MSE↓/SSIM↑)   평균  "
    else:
        _langname = {"en": "English", "hard": "extreme fonts"}.get(args.lang, "Korean")
        _title = f"{_langname} VAE reconstruction (roundtrip encode->decode, MSE lower / SSIM higher)   mean  "
    base = float(np.mean(agg[vaes[0][0]]["m"]))
    seg = []
    for lbl, _ in vaes:
        mm = float(np.mean(agg[lbl]["m"])); ss = float(np.mean(agg[lbl]["s"]))
        if lbl == vaes[0][0]:
            seg.append(f"{lbl} {mm:.4f}/{ss:.3f}")
        else:
            d = (mm / base - 1) * 100
            seg.append(f"{lbl} {mm:.4f}/{ss:.3f}({'+' if d > 0 else ''}{d:.0f}%)")
    summ = Image.new("L", (Wm, 30), 255)
    ImageDraw.Draw(summ).text((6, 6), _title + "  |  ".join(seg), font=hdrf, fill=0)
    out = np.vstack([np.array(summ), np.full((3, Wm), 0, np.uint8)] + blocks)
    out = np.array(Image.fromarray(out).resize((Wm * 2, out.shape[0] * 2), Image.NEAREST))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})")
    print("  |  ".join(seg))


if __name__ == "__main__":
    main()
