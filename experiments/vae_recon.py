"""기존 VAE vs 학습(한글적응) VAE 의 **한글 복원 능력** 비교 이미지.

순수 VAE roundtrip(encode.mode → decode) 만 본다(Eruku 생성 무관). 한 텍스트마다
[원본 | 기존VAE 복원 | 학습VAE 복원] 을 세로로 쌓고 각 복원의 MSE↓/SSIM↑ 를 표시.
한글은 밀집 2D 음절이라 기존 VAE 가 뭉개던 게(특히 작고 빽빽한 글자) 학습 VAE 에서 얼마나
선명해지는지 눈+숫자로 확인.

예:
  CUDA_VISIBLE_DEVICES=2 python vae_recon.py \
    --vae-new finetune_runs/vae_korean_dec/vae_s15000 \
    --out finetune_runs/korean_en2ko_fixed/_eval/recon_ko_old_vs_new.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
from diffusers import AutoencoderKL

from common import (REPO, TRAIN_FONTS as TRAIN, TEST_FONTS as UNSEEN, load_line_x11,
                     roundtrip, metrics, to_u8, label_fonts, strip_row, pad_blocks, vae_summary)
from infer.show import render_in_font

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help='"라벨=경로" 목록. 각 텍스트마다 원본 아래로 이 순서대로 복원 stack. '
                         '예: 원본=blowing-up-groundhogs/emuru_vae decoder=.../vae_dec/vae_s15000 full=.../vae_full/vae_s15000')
    ap.add_argument("--out", default=str(REPO / "finetune_runs/korean_en2ko_fixed/_eval/recon_ko.png"))
    ap.add_argument("--lang", choices=["ko", "en", "hard"], default="ko", help="ko=한글 / en=영어 / hard=극단폰트")
    ap.add_argument("--label-lang", choices=["ko", "en"], default="ko",
                    help="figure 안 고정 라벨/제목 언어. en = 공개(HF 모델카드) 용")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    samples = {"en": EN_SAMPLES, "hard": HARD_SAMPLES}.get(args.lang, SAMPLES)

    specs = [m.split("=", 1) for m in args.models]        # [(label, path), ...]
    vaes = [(lbl, load_vae(path, dev)) for lbl, path in specs]
    lab, hdrf = label_fonts()

    LW = 110                                               # 좌측 라벨 폭
    blocks = []
    agg = {lbl: {"m": [], "s": []} for lbl, _ in vaes}     # 모델별 MSE/SSIM 누적
    for font, text in samples:
        if not Path(font).exists():
            print(f"[skip] font 없음 {font}"); continue
        x = load_line_x11(render_in_font(font, text), h=H)
        w = x.shape[-1]
        head = Image.new("L", (LW + w, 24), 235)
        ImageDraw.Draw(head).text((6, 3), f"[{Path(font).stem}] {text}", font=hdrf, fill=0)
        parts = [np.array(head),
                 strip_row(to_u8(x), "원본" if args.label_lang == "ko" else "source", w, LW, H, lab)]
        for lbl, vae in vaes:
            r = roundtrip(vae, x, dev)
            m, s = metrics(x, r)
            agg[lbl]["m"].append(m); agg[lbl]["s"].append(s)
            parts += [np.full((2, LW + w), 210, np.uint8),
                      strip_row(to_u8(r), lbl, w, LW, H, lab, f"MSE {m:.4f} SSIM {s:.3f}")]
        parts.append(np.full((8, LW + w), 90, np.uint8))
        blocks.append(np.vstack(parts))

    Wm, blocks = pad_blocks(blocks)
    # 요약 헤더: 모델별 평균 MSE/SSIM + 첫 모델(보통 원본) 대비 증감
    if args.label_lang == "ko":
        _langname = {"en": "영어", "hard": "극단폰트"}.get(args.lang, "한글")
        _title = f"{_langname} VAE 복원 비교 (roundtrip, MSE↓/SSIM↑)   평균  "
    else:
        _langname = {"en": "English", "hard": "extreme fonts"}.get(args.lang, "Korean")
        _title = f"{_langname} VAE reconstruction (roundtrip encode->decode, MSE lower / SSIM higher)   mean  "
    seg = vae_summary(agg, [lbl for lbl, _ in vaes])
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
