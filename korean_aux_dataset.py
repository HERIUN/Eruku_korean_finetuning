"""한글 aux 학습(HTR finetune, VAE finetune) 공용 데이터 어댑터.

원본 emuru VAE 는 `rgb(styled 3ch 입력) → bw(clean 1ch 타깃)` 디노이징 AE 이고, HTR 은 `bw→text`.
이 어댑터는 한 폰트·한 텍스트에서 **픽셀 정렬된 (rgb, bw) 페어** + 라벨/마스크를 만든다:
  - geometry(리사이즈)는 rgb·bw 공유 → 픽셀 정렬(디노이징 L1 유효)
  - appearance(배경합성·블러·jitter)는 **rgb 에만** → styled 입력 / clean 타깃

산출(고정 64×768, HTR/VAE 가 이 해상도로 학습됨):
  rgb [3,64,768] [-1,1], bw [1,64,768] [-1,1], writer_id(int, font idx),
  text_logits_s2s [T]([sos]+chars+[eos]), texts_len(=eos 인덱스)
collate:
  text_logits_s2s [B,T](pad=0), texts_len [B], tgt_key_padding_mask [B,T](True=pad),
  tgt_key_mask = subsequent_mask(T-1)  (decoder self-attn causal, 입력이 [:, :-1] 라 T-1)

렌더 파이프라인/샘플러는 korean_split_dataset.make_dataset(KoreanSplitFontSquare) 재사용.
알파벳은 custom_datasets.korean_alphabet(코퍼스 기반, latin 인덱스 pretrained 보존).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

from korean_split_dataset import make_dataset
from custom_datasets.korean_alphabet import get_korean_alphabet
from custom_datasets.subsequent_mask import subsequent_mask

H = 64
W = 768


def _load_backgrounds():
    """assets/backgrounds 의 배경(3ch [0,1] 텐서) 리스트. 없으면 흰 배경 1개."""
    bgs = []
    d = ASSETS / "backgrounds"
    if d.is_dir():
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                im = Image.open(p).convert("RGB")
                bgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
    if not bgs:
        bgs = [torch.ones(3, 256, 256)]
    return bgs


def _bg_patch(bg, h, w, rng: random.Random):
    """배경 텐서에서 [3,h,w] 패치(랜덤 크롭, 부족하면 resize)."""
    _, bh, bw = bg.shape
    if bh >= h and bw >= w:
        y0 = rng.randint(0, bh - h)
        x0 = rng.randint(0, bw - w)
        return bg[:, y0:y0 + h, x0:x0 + w]
    return F.interpolate(bg.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]


def _fit_width(img):
    """[C,64,w] → [C,64,768]: w<768 오른쪽 흰(1.0) 패딩, w>768 이면 폭 스케일-투-핏(라벨 정합 유지)."""
    c, h, w = img.shape
    if w == W:
        return img
    if w < W:
        out = img.new_ones((c, h, W))
        out[:, :, :w] = img
        return out
    return F.interpolate(img.unsqueeze(0), size=(h, W), mode="bilinear", align_corners=False)[0]


class KoreanAuxDataset(Dataset):
    def __init__(self, length, style_range=(1, 8), gen_range=(1, 12), seed=42,
                 fonts_dir=None, blur_p=0.5, jitter_p=0.5, alpha_range=(0.75, 1.0),
                 augment=False, uniform_freq=False):
        # render_tf/샘플러/renderers 재사용 (KoreanSplitFontSquare)
        self.base = make_dataset(style_range=style_range, gen_range=gen_range,
                                 length=length, seed=seed, fonts_dir=fonts_dir)
        self.render_tf = self.base.render_tf
        self.samplers = [self.base.style_sampler, self.base.gen_sampler]
        self.n_fonts = self.base.renderers_length
        self.alpha = get_korean_alphabet()
        self.in_alpha = set(self.alpha.char2idx)
        self.bgs = _load_backgrounds()
        self.length = length
        self.blur_p, self.jitter_p, self.alpha_range = blur_p, jitter_p, alpha_range
        self.augment = augment   # shape 증강(획 morph + 소각 회전 + elastic warp) = 스타일 다양성 합성. 학습에만.
        self.uniform_freq = uniform_freq  # 논문식 균일 음절빈도(모든 글리프 균등 노출)
        # 균일빈도 텍스트용 문자 풀
        self._ko = [c for c in self.in_alpha if 0xAC00 <= ord(c) <= 0xD7A3]
        self._lat = [c for c in self.in_alpha if c.isascii() and c.isalpha()]
        self._dig = list("0123456789")

    def __len__(self):
        return self.length

    def _uniform_text(self, rng):
        """논문 'character frequency almost uniform': 코퍼스빈도 대신 균등 랜덤 음절로 텍스트 구성.
        VAE 는 shape 만 복원하므로 실제 문장일 필요 없음 → 모든 글리프 균등 노출로 커버리지↑."""
        words = []
        for _ in range(rng.randint(1, 12)):
            w = []
            for _ in range(rng.randint(1, 5)):
                r = rng.random()
                w.append(rng.choice(self._ko) if r < 0.7 else
                         rng.choice(self._lat) if r < 0.9 else rng.choice(self._dig))
            words.append("".join(w))
        return " ".join(words)

    def _render_base(self, rng):
        """텍스트+폰트 렌더 → (base64 [1,64,w] float[0,1] ink=0/bg=1, label 문자열). 비어있으면 재시도."""
        for _ in range(8):
            font_id = rng.randrange(self.n_fonts)
            text = self._uniform_text(rng) if self.uniform_freq else rng.choice(self.samplers)()
            text = "".join(c for c in text if c in self.in_alpha)  # 알파벳 밖 문자 제거(이미지↔라벨 정합)
            if not text.strip():
                continue
            s = self.render_tf({"text": text, "font_id": font_id})
            label = s["text"]  # 실제 렌더된(폰트 charset 필터 반영) 텍스트
            label = "".join(c for c in label if c in self.in_alpha)
            if not label.strip():
                continue
            img = s["img"]  # [1,Hs,Ws] float, ink≈0 bg≈1
            _, hs, ws = img.shape
            w64 = max(1, int(round(H * ws / hs)))            # 높이 64 로 리사이즈, 종횡비 유지
            base64 = F.interpolate(img.unsqueeze(0), size=(H, w64), mode="bilinear", align_corners=False)[0]
            base64 = base64.clamp(0, 1)
            return base64, label, font_id
        # fallback: 공백 한 칸
        return torch.ones(1, H, 8), " ", 0

    def _make_rgb(self, base64, rng):
        """clean base64([1,64,w] ink=0/bg=1) → styled rgb [3,64,w] [-1,1]."""
        _, h, w = base64.shape
        text_img = 1.0 - base64                              # ink=1, bg=0
        alpha = rng.uniform(*self.alpha_range)
        bg = _bg_patch(self.bgs[rng.randrange(len(self.bgs))], h, w, rng)  # [3,64,w] [0,1]
        merged = (1.0 - text_img * alpha) * bg               # ink=alpha 어둠, bg=배경
        if rng.random() < self.blur_p:
            k = rng.choice([3, 3, 5])
            merged = torchvision_gaussian_blur(merged, k)
        if rng.random() < self.jitter_p:
            b = rng.uniform(0.85, 1.15); merged = (merged * b).clamp(0, 1)      # brightness
            c = rng.uniform(0.9, 1.1); m = merged.mean(); merged = ((merged - m) * c + m).clamp(0, 1)  # contrast
        return merged * 2.0 - 1.0                            # [-1,1]

    def _augment_shape(self, x, rng):
        """clean base64([1,64,w] ink=0/bg=1) 에 morph(획굵기)+회전+elastic warp → 스타일 다양성 합성.
        논문 font_square 식 스타일 증폭. target(bw)·input(rgb) 이 같은 augmented shape 공유하도록 base 에 적용."""
        t = x.unsqueeze(0)                                   # [1,1,64,w]
        if rng.random() < 0.6:                               # morph: 획 굵기
            if rng.random() < 0.5:
                t = -F.max_pool2d(-t, 3, stride=1, padding=1)  # dilate: 획 굵게
            else:
                t = F.max_pool2d(t, 3, stride=1, padding=1)    # erode: 획 얇게
        if rng.random() < 0.5:                               # elastic warp: 글리프 형태 변형(폰트 다양성 모사)
            t = self._elastic(t, rng)
        t = t[0]
        if rng.random() < 0.4:                               # 소각 회전
            from torchvision.transforms.functional import rotate
            t = rotate(t, rng.uniform(-3, 3), fill=1.0)
        return t.clamp(0, 1)

    @staticmethod
    def _elastic(t, rng, amp=0.025, grid=6):
        """coarse 랜덤 변위장 grid_sample = TPS 유사 탄성 변형(획 형태를 부드럽게 왜곡)."""
        _, _, h, w = t.shape
        g = torch.manual_seed(rng.randrange(1 << 30))        # 재현 무관, 매번 랜덤
        dx = torch.randn(1, 1, grid, grid) * amp
        dy = torch.randn(1, 1, grid, grid) * amp
        dx = F.interpolate(dx, size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        dy = F.interpolate(dy, size=(h, w), mode="bilinear", align_corners=True)[0, 0]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        grid_t = torch.stack([xs + dx, ys + dy], -1).unsqueeze(0)
        return F.grid_sample(t, grid_t, mode="bilinear", padding_mode="border", align_corners=True)

    def __getitem__(self, i):
        rng = random.Random()  # 온라인 무한 렌더라 재현성 불필요
        base64, label, font_id = self._render_base(rng)
        if self.augment:
            base64 = self._augment_shape(base64, rng)
        bw = _fit_width(base64 * 2.0 - 1.0)                  # [1,64,768] [-1,1] clean
        rgb = _fit_width(self._make_rgb(base64, rng))        # [3,64,768] [-1,1] styled
        chars = [c for c in label if c in self.in_alpha]
        tokens = [self.alpha.sos] + self.alpha.encode(chars).tolist() + [self.alpha.eos]
        return {
            "rgb": rgb, "bw": bw, "writer_id": int(font_id),
            "tokens": tokens, "eos_index": len(chars) + 1,  # NoisyTeacher 의 unpadded_text_len
        }


def torchvision_gaussian_blur(img, kernel):
    from torchvision.transforms.functional import gaussian_blur
    return gaussian_blur(img, kernel_size=kernel)


def aux_collate(batch):
    B = len(batch)
    T = max(len(b["tokens"]) for b in batch)
    text = torch.zeros(B, T, dtype=torch.long)              # pad=0
    for i, b in enumerate(batch):
        text[i, :len(b["tokens"])] = torch.tensor(b["tokens"], dtype=torch.long)
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),       # [B,3,64,768]
        "bw": torch.stack([b["bw"] for b in batch]),         # [B,1,64,768]
        "writer_id": torch.tensor([b["writer_id"] for b in batch], dtype=torch.long),
        "text_logits_s2s": text,                             # [B,T]
        "texts_len": torch.tensor([b["eos_index"] for b in batch], dtype=torch.long),
        "tgt_key_padding_mask": (text == 0),                 # [B,T] True=pad
        "tgt_key_mask": subsequent_mask(T - 1),              # [T-1,T-1] causal (입력=text[:,:-1])
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="/tmp/kaux")
    args = ap.parse_args()
    from torch.utils.data import DataLoader
    ds = KoreanAuxDataset(length=args.n)
    dl = DataLoader(ds, batch_size=args.n, collate_fn=aux_collate, num_workers=0)
    b = next(iter(dl))
    for k, v in b.items():
        print(k, tuple(v.shape) if torch.is_tensor(v) else v)
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        rgb = ((b["rgb"][i].mean(0) + 1) / 2 * 255).byte().numpy()
        bw = ((b["bw"][i, 0] + 1) / 2 * 255).byte().numpy()
        sep = np.full((3, rgb.shape[1]), 128, np.uint8)
        txt = ds.alpha._decode(b["text_logits_s2s"][i], ds.alpha.extra_tokens)
        Image.fromarray(np.vstack([rgb, sep, bw])).save(outd / f"{i:02d}.png")
        print(f"{i}: writer={b['writer_id'][i].item()} len={b['texts_len'][i].item()} text={txt!r}")
    print("saved", outd)
