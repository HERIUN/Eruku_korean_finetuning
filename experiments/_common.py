"""experiments/ 공용 헬퍼 — repo 루트 부트스트랩 + 실험 스크립트들이 반복하던 로직.

이 폴더의 스크립트는 `from _common import …` 만으로 repo 루트가 sys.path 에 들어가
루트 모듈(infer_show, eval_htr_cer …)을 그대로 import 할 수 있다
(PYTHONPATH 불필요, 어느 cwd 에서든 `python experiments/xxx.py` 로 실행 가능).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HW_DIR = REPO / "custom_datasets" / "hw_line_sample"
LABEL_FONT = REPO / "assets" / "fonts_label" / "NanumGothic-Regular.ttf"
TRAIN_FONTS = REPO / "assets" / "fonts_korean_v2" / "train"
TEST_FONTS = REPO / "assets" / "fonts_korean_v2" / "test"
HTR_CKPT = REPO / "finetune_runs" / "aux_htr_ko" / "htr_s20000"


def read_hw_labels(d=None):
    """hw_line_sample/label.txt → [(png Path, 전사)] (존재하는 파일만, label.txt 순서=이름순)."""
    d = Path(d) if d else HW_DIR
    rows = []
    for ln in (d / "label.txt").read_text(encoding="utf-8").splitlines():
        if "\t" in ln:
            fn, txt = ln.split("\t", 1)
            p = d / fn.strip()
            if p.exists():
                rows.append((p, txt.strip()))
    return rows


def load_line_x11(src, h=64, pad8=True):
    """라인이미지(경로 또는 grayscale 배열) → [1,1,h,W] [-1,1] (bg=+1, ink=-1).

    bilinear h-리사이즈, pad8=True 면 폭 8배수 흰(+1) 우측패딩(VAE 정확 roundtrip 조건).
    """
    if isinstance(src, (str, Path)):
        a = np.array(Image.open(src).convert("L"))
    else:
        a = np.asarray(src)
    t = torch.from_numpy(a.astype(np.float32) / 255.0)[None, None]
    w = max(1, int(round(h * a.shape[1] / a.shape[0])))
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    x = t * 2 - 1
    w8 = (w + 7) // 8 * 8
    if pad8 and w8 != w:
        x = F.pad(x, (0, w8 - w), value=1.0)
    return x


@torch.no_grad()
def roundtrip(vae, x, dev):
    """[1,1,h,W] [-1,1] → VAE encode.mode→decode → [1,1,h,W] [-1,1]."""
    z = vae.encode(x.repeat(1, 3, 1, 1).to(dev)).latent_dist.mode()
    return vae.decode(z).sample.clamp(-1, 1)[:, :1].cpu()


def metrics(orig, rec):
    """[-1,1] 텐서 쌍 → (MSE↓, SSIM↑) ([0,1] 도메인 기준)."""
    a = ((orig[0, 0] + 1) / 2).numpy(); b = ((rec[0, 0] + 1) / 2).numpy()
    return float(np.mean((a - b) ** 2)), float(ssim(a, b, data_range=1.0))


def to_u8(t):
    return ((t[0, 0] + 1) / 2 * 255).clamp(0, 255).byte().numpy()


def read_cer(htr, arr_u8, text, dev):
    """grayscale uint8 라인이미지를 한글 HTR 리더로 읽어 text 와의 CER."""
    from eval_htr_cer import to_htr_input, htr_read, norm
    from train_aux_htr import cer as char_cer
    return char_cer([norm(htr_read(htr, to_htr_input(arr_u8, dev), dev)[0])], [norm(text)])


# ── montage 헬퍼 (vae 복원 비교류: 원본 아래 모델별 복원 stack) ─────────────────

def label_fonts(size_small=15, size_head=17):
    """(본문 라벨, 헤더) PIL 폰트 쌍."""
    if LABEL_FONT.exists():
        return (ImageFont.truetype(str(LABEL_FONT), size_small),
                ImageFont.truetype(str(LABEL_FONT), size_head))
    return ImageFont.load_default(), ImageFont.load_default()


def strip_row(t_u8, name, w, lw, h, font, extra=""):
    """[h,W] 이미지 좌측에 폭 lw 라벨(name, 하단 extra)을 붙인 한 줄 [h, lw+w]."""
    row = np.full((h, lw + w), 255, np.uint8)
    row[:, lw:lw + t_u8.shape[1]] = t_u8
    im = Image.fromarray(row)
    d = ImageDraw.Draw(im)
    d.text((6, 4), name, font=font, fill=0)
    if extra:
        d.text((6, h - 22), extra, font=font, fill=0)
    return np.array(im)


def pad_blocks(blocks):
    """세로 블록들을 최대 폭에 맞춰 흰 패딩 → (최대폭, 블록 리스트)."""
    wm = max(b.shape[1] for b in blocks)
    return wm, [np.hstack([b, np.full((b.shape[0], wm - b.shape[1]), 255, np.uint8)])
                if b.shape[1] < wm else b for b in blocks]


def vae_summary(agg, order):
    """모델별 평균 MSE/SSIM 요약 세그먼트(첫 모델 대비 MSE 증감 %)."""
    base = float(np.mean(agg[order[0]]["m"]))
    seg = []
    for lbl in order:
        mm = float(np.mean(agg[lbl]["m"])); ss = float(np.mean(agg[lbl]["s"]))
        if lbl == order[0]:
            seg.append(f"{lbl} {mm:.4f}/{ss:.3f}")
        else:
            d = (mm / base - 1) * 100
            seg.append(f"{lbl} {mm:.4f}/{ss:.3f}({'+' if d > 0 else ''}{d:.0f}%)")
    return seg


# ── montage 헬퍼 (생성 비교류: infer_show.cell 그리드) ─────────────────────────

def header_row(col_labels, cw, ch):
    """컬럼 라벨 헤더 한 줄."""
    from infer_show import cell
    return np.hstack([cell(np.full((ch, cw), 245, np.uint8), lbl, cw, ch) for lbl in col_labels])


def fit_row(cells, full_w):
    """셀들을 가로로 붙이고 full_w 까지 흰 패딩."""
    row = np.hstack(cells)
    if row.shape[1] < full_w:
        row = np.hstack([row, np.full((row.shape[0], full_w - row.shape[1]), 255, np.uint8)])
    return row
