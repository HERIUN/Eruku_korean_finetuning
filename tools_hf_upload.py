"""HF 릴리즈 도구 — 학습 ckpt 를 HuggingFace 공개 repo 2개로 발행.

    HERIUN/eruku_korean       상위 모델(T5 + proj + 특수토큰) + remote code(modeling/configuration)
    HERIUN/emuru_vae_korean   한글 적응 VAE (diffusers AutoencoderKL)

**왜 VAE 를 따로 올리나**: `modeling_eruku.py` 는
`AutoencoderKL.from_pretrained(config.vae_name_or_path)` 로 **VAE 가중치를 그 repo 에서
실제로 내려받는다**. 우리 최고 조합은 `--train-part full` 로 학습해 latent 공간이 이동한
한글 VAE 를 전제하므로, config 가 원본 영어 VAE(`blowing-up-groundhogs/emuru_vae`)를
가리키는 채로 T5 가중치만 올리면 조합이 깨져 생성이 무너진다.
반면 **T5 는 별도 업로드가 필요 없다** — `T5Config.from_pretrained(config.t5_name_or_path)`
로 구조(하이퍼파라미터)만 받고, 가중치는 상위 repo 의 weight 파일에서 전부 덮인다.

사용법:
    # 1) VAE 먼저 (상위 repo 로드가 이걸 필요로 함).
    #    --figures-dir 를 주면 그 안의 before/after figure 를 samples/ 로 올리고 카드에 삽입
    #    (figure 만드는 명령은 README 「HuggingFace 릴리즈」 절 0) 단계)
    .venv/bin/python tools_hf_upload.py vae --figures-dir /tmp/relfig --push

    # 2) 상위 모델: 변환 → 키 정합성 assert → (선택) 공개 API 로 CER 검증 → 업로드
    .venv/bin/python tools_hf_upload.py model --verify-cer             # dry-run
    .venv/bin/python tools_hf_upload.py model --figures-dir /tmp/relfig --push

기본은 **dry-run**(로컬 스테이징 디렉토리만 생성). `--push` 를 줄 때만 실제 업로드하며
토큰은 `HF_TOKEN` env 또는 `hf auth login` 캐시를 쓴다.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# transformers 의 remote-code 로더는 HF_MODULES_CACHE 아래로 모듈을 복사해 import 한다.
# 기본 경로(~/.cache/huggingface/modules)가 다른 사용자 소유면 PermissionError 가 나므로
# transformers import 전에(=여기서) 쓰기 가능한 경로로 고정한다.
os.environ.setdefault("HF_MODULES_CACHE", f"/tmp/hf_modules_{os.getuid()}")

DEF_CKPT = HERE / "finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth"
DEF_VAE_SRC = HERE / "finetune_runs/vae_korean_full_mse/vae_s28000"
DEF_MODEL_REPO = "HERIUN/eruku_korean"
DEF_VAE_REPO = "HERIUN/emuru_vae_korean"
UPSTREAM_VAE = "blowing-up-groundhogs/emuru_vae"
UPSTREAM_MODEL = "blowing-up-groundhogs/eruku"

# 아키텍처를 결정하는 키 — 이게 upstream 과 다르면 latent 규약이 깨진 것이므로 abort
VAE_ARCH_KEYS = ("in_channels", "out_channels", "latent_channels", "sample_size",
                 "block_out_channels", "layers_per_block", "down_block_types", "up_block_types",
                 "norm_num_groups", "act_fn")


# ─────────────────────────────────────────────────────────────────────────────
# modeling_eruku.py 패치: 실제 손글씨 style 의 여백 제거(ink bbox 크롭)
# ─────────────────────────────────────────────────────────────────────────────
# 근거: docs/VAE_ROBUSTNESS.md ② — 스캔 여백/테두리가 VAE latent std 를 폰트 대비 6~29배로
# 튀겨 T5 를 OOD 로 밀고(runaway/blank) 생성이 무너진다. 학습이 style 을 항상 프레임 꽉 찬
# 상태로만 봤기 때문. 추론 단계 크롭으로 재학습 없이 해결된다(_eval_mirror_compare.py 검증).
BBOX_HELPER = '''

def ink_bbox_crop(image: "Image.Image", thr: int = 180, bthr: int = 215,
                  dens_col: float = 0.03, dens_row: float = 0.006,
                  solid: float = 0.55, pad: int = 4) -> "Image.Image":
    """Crop a style image tight to its ink bounding box.

    Real scanned handwriting usually comes with page margins or a scan frame.
    Those empty regions blow up the VAE latent std (6-29x vs. font renders),
    pushing the T5 decoder out of distribution -> runaway / blank generation.
    Training only ever saw style crops that fill the frame, so we close that
    gap at inference time.

    Two stages: (1) peel solid border rows/columns (scan frame), (2) density
    based bounding box. The row threshold is deliberately lower than the column
    one so sparse strokes (Hangul jongseong, descenders) are not clipped.
    """
    a = np.array(image.convert("L"))
    if a.size == 0:
        return image
    bink = a < bthr                                    # border detection (incl. grey)
    y0, y1, x0, x1 = 0, a.shape[0], 0, a.shape[1]
    while y0 < y1 and bink[y0].mean() > solid: y0 += 1
    while y1 > y0 and bink[y1 - 1].mean() > solid: y1 -= 1
    while x0 < x1 and bink[:, x0].mean() > solid: x0 += 1
    while x1 > x0 and bink[:, x1 - 1].mean() > solid: x1 -= 1
    if y1 - y0 < 2 or x1 - x0 < 2:
        return image
    inner = a[y0:y1, x0:x1]
    ink = inner < thr                                  # text density
    cols = np.where(ink.mean(0) > dens_col)[0]
    rows = np.where(ink.mean(1) > dens_row)[0]
    if len(cols) == 0 or len(rows) == 0:
        return image.crop((x0, y0, x1, y1))
    yy0, yy1 = max(0, int(rows.min()) - pad), min(inner.shape[0], int(rows.max()) + pad + 1)
    xx0, xx1 = max(0, int(cols.min()) - pad), min(inner.shape[1], int(cols.max()) + pad + 1)
    return image.crop((x0 + xx0, y0 + yy0, x0 + xx1, y0 + yy1))

'''

PATCH_ANCHOR_CLASS = "class ErukuPreTrainedModel(PreTrainedModel):"

PATCH_SIG_OLD = """        style_text: str = "",
        cfg_scale: float = 1.25,"""
PATCH_SIG_NEW = """        style_text: str = "",
        cfg_scale: float = 1.25,
        bbox_crop: bool = True,"""

PATCH_DOC_OLD = """            cfg_scale: Classifier-free guidance scale (default: 1.25)"""
PATCH_DOC_NEW = """            cfg_scale: Classifier-free guidance scale (default: 1.25)
            bbox_crop: Crop the style image tight to its ink bounding box first
                (default: True). Recommended for real scans -- page margins push
                the VAE latents out of distribution and the model then runs away
                or returns blanks. Set False for style crops that already fill
                the frame (e.g. font renders)."""

PATCH_BODY_OLD = """        # Preprocess style image
        style_img = style_image.convert('RGB')"""
PATCH_BODY_NEW = """        # Preprocess style image
        if bbox_crop:
            style_image = ink_bbox_crop(style_image)
        style_img = style_image.convert('RGB')"""


def patch_modeling(src: str) -> str:
    """modeling_eruku.py 소스에 ink-bbox 크롭을 주입. 앵커가 하나라도 없으면 예외."""
    if "def ink_bbox_crop" in src:
        raise SystemExit("[patch] 이미 패치된 modeling_eruku.py 입니다 (ink_bbox_crop 존재)")
    for anchor in (PATCH_ANCHOR_CLASS, PATCH_SIG_OLD, PATCH_DOC_OLD, PATCH_BODY_OLD):
        if src.count(anchor) != 1:
            raise SystemExit(f"[patch] 앵커를 정확히 1번 찾지 못함 (n={src.count(anchor)}):\n{anchor}")
    src = src.replace(PATCH_ANCHOR_CLASS, BBOX_HELPER.strip("\n") + "\n\n\n" + PATCH_ANCHOR_CLASS)
    src = src.replace(PATCH_SIG_OLD, PATCH_SIG_NEW)
    src = src.replace(PATCH_DOC_OLD, PATCH_DOC_NEW)
    src = src.replace(PATCH_BODY_OLD, PATCH_BODY_NEW)
    return src


# ─────────────────────────────────────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────────────────────────────────────
def _stage_figures(out: Path, figdir: Path, wanted: dict):
    """figdir 의 figure 를 out/samples/ 로 복사. wanted = {카드에서 쓸 키: 파일명}.
    없는 파일은 조용히 건너뛰고, 실제로 복사된 것만 {키: repo 내 경로} 로 돌려준다."""
    got = {}
    if not figdir or not figdir.exists():
        return got
    dst = out / "samples"
    for key, name in wanted.items():
        src = figdir / name
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst / name)
            got[key] = f"samples/{name}"
            print(f"[fig] samples/{name}  {src.stat().st_size/1e6:.2f}MB")
        else:
            print(f"[fig] 없음(건너뜀) {src}")
    return got


def _url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{path}"


def _api(token=None):
    from huggingface_hub import HfApi
    tok = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)
    who = api.whoami()
    print(f"[hf] 인증 OK: {who.get('name')} ({who.get('type')})")
    return api


def _push(api, folder: Path, repo_id: str, msg: str, delete_patterns=None):
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    url = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                            commit_message=msg, delete_patterns=delete_patterns)
    print(f"[hf] 업로드 완료 → https://huggingface.co/{repo_id}\n     {url}")


# ─────────────────────────────────────────────────────────────────────────────
# vae 서브커맨드
# ─────────────────────────────────────────────────────────────────────────────
VAE_CARD = """---
license: mit
library_name: diffusers
base_model: blowing-up-groundhogs/emuru_vae
base_model_relation: finetune
language:
- ko
- en
tags:
- vae
- autoencoder
- handwriting-generation
- korean
---

# emuru_vae_korean

Korean-adapted VAE for [Eruku](https://github.com/Blowing-Up-Groundhogs/Eruku) styled
handwriting generation, fine-tuned from
[`blowing-up-groundhogs/emuru_vae`](https://huggingface.co/blowing-up-groundhogs/emuru_vae).

The original VAE was trained on Latin script and is the **hard ceiling** for Korean
fidelity: Eruku conditions and predicts only VAE latents (8 dims per latent column), so
anything `decode(encode(x))` cannot represent is invisible to the model. Hangul stacks
initial/medial/final jamo in 2D inside one syllable cell, which costs ~12x more
reconstruction error than Latin. This checkpoint removes most of that gap.

## Reconstruction (roundtrip encode -> decode, fixed probe)

| | Korean MSE | Korean SSIM | Extreme fonts MSE | English MSE |
|---|---|---|---|---|
| `blowing-up-groundhogs/emuru_vae` | {ORIG_KO_MSE} | {ORIG_KO_SSIM} | {ORIG_HARD_MSE} | {ORIG_EN_MSE} |
| **this model** | **{NEW_KO_MSE}** | **{NEW_KO_SSIM}** | **{NEW_HARD_MSE}** | **{NEW_EN_MSE}** |

English improves too, because the recipe is script-neutral (pure L1, no OCR loss).
{FIGURES}
## Recipe

```
--train-part full --htr-weight 0 --kl-weight 1e-6 --batch-size 8 --grad-accum 4 --lr 1e-4
```
Pure L1 reconstruction (+ negligible KL), encoder and decoder both unfrozen, warm-started
through a decoder-only stage and then a 15k-step full stage; this checkpoint is 28k further
steps on top of that. Training data: synthetic Korean/English line renders
from 83 Korean + 177 Latin fonts (fonts act as writers), held-out font split for
validation. See [`train_vae_korean.py`](https://github.com/HERIUN/Eruku_korean_finetuning).

Adding an OCR/HTR readability loss made things **worse** for this task (Korean MSE
0.0089 vs 0.0031 with it off) — text is already high contrast, so L1 keeps strokes crisp
on its own, while a Korean HTR loss biases the shared decoder and degrades English.

## Usage

Copy these two helpers and you never have to think about the input conventions again
(they are inherited from `emuru_vae`, not chosen by us):

```python
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np, torch, torch.nn.functional as F

def to_vae_input(image, height=64):
    # PIL text-line image -> [1, 3, height, W] in [-1, 1], width padded to a multiple of 8
    g = image.convert("L")
    w = max(1, round(height * g.width / g.height))
    x = torch.from_numpy(np.array(g, dtype=np.float32) / 255.0)[None, None]
    x = F.interpolate(x, size=(height, w), mode="bilinear", align_corners=False).clamp(0, 1)
    x = F.pad(x * 2 - 1, (0, -w % 8), value=1.0)     # [-1,1], white pad
    return x.repeat(1, 3, 1, 1)                       # encoder wants 3 channels

def to_pil(t):
    # VAE output [1, 1, H, W] in [-1, 1] -> grayscale PIL image
    return Image.fromarray((((t[0, 0].clamp(-1, 1) + 1) / 2) * 255).byte().cpu().numpy())
```

Then a roundtrip is three lines:

```python
vae = AutoencoderKL.from_pretrained("{VAE_REPO}").eval().cuda()
x = to_vae_input(Image.open("line.png"))

with torch.no_grad():
    z = vae.encode(x.cuda()).latent_dist.mode()      # [1, 1, 8, W/8]  (.sample() adds noise)
    rec = vae.decode(z).sample[:, :1]                # decoder returns 1 channel

to_pil(rec).save("line_recon.png")
print(tuple(x.shape), "->", tuple(z.shape), "->", tuple(rec.shape))
# (1, 3, 64, 872) -> (1, 1, 8, 109) -> (1, 1, 64, 872)
```

One latent column covers 8 pixels and carries 8 dims — that column sequence is exactly what
Eruku conditions on and predicts.

### Which of those conventions actually matter

Measured by breaking one at a time on the same input line:

| convention | what happens if you ignore it |
|---|---|
| 3 input channels | **hard error** (`expected input[1, 1, 64, 872] to have 3 channels`) |
| values in [-1, 1] | runs and looks plausible, but the latents are wrong — output shifts by MSE 0.010 and Eruku conditioning degrades silently. The easiest mistake to make |
| height 64 | runs, but the latent height follows the input (96px -> 12 rows), so it no longer matches Eruku's 8-dim-per-column interface |
| width a multiple of 8 | runs; the output is silently up to 7px narrower than the input (871 -> 864) |

To generate handwriting rather than reconstruct it, use the paired model
[`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO}) — it loads this VAE automatically and
does the preprocessing for you.

The metrics above the figures are measured on clean font renders. Real scanned handwriting
reconstructs noticeably worse (roundtrip MSE ~0.016 on our 20-line scan probe): the decoder
snaps ink to crisp black and does not preserve faint or pencil strokes.

> **Warning — latent space moved.** This was trained with `--train-part full`
> (encoder + decoder), so its latents are **not interchangeable** with the original
> `emuru_vae`. Pairing it with the original `blowing-up-groundhogs/eruku` weights will
> produce garbage. Use it with [`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO}),
> which was re-adapted on top of these latents.

## Limitations

Held-out extreme display/brush fonts remain considerably worse than regular fonts
({NEW_HARD_MSE} vs {NEW_KO_MSE}); font diversity is the bottleneck (99 fonts here vs
~100k upstream), and augmentation did not help.

License: MIT, inherited from the base model.
"""


def cmd_vae(args):
    src = Path(args.src)
    out = Path(args.out)
    cfg_p = src / "config.json"
    w_p = src / "diffusion_pytorch_model.safetensors"
    for p in (cfg_p, w_p):
        if not p.exists():
            raise SystemExit(f"[vae] 없음: {p}")

    # upstream 과 아키텍처 동일성 확인 (latent 규약이 깨졌으면 발행 중단)
    from huggingface_hub import hf_hub_download
    up = json.loads(Path(hf_hub_download(UPSTREAM_VAE, "config.json")).read_text())
    cfg = json.loads(cfg_p.read_text())
    diff = {k: (up.get(k), cfg.get(k)) for k in VAE_ARCH_KEYS if up.get(k) != cfg.get(k)}
    if diff:
        raise SystemExit(f"[vae] upstream 과 아키텍처 불일치 → abort: {diff}")
    print(f"[vae] 아키텍처 upstream 동일 확인 ({len(VAE_ARCH_KEYS)}키)")

    out.mkdir(parents=True, exist_ok=True)
    cfg["_name_or_path"] = args.repo            # 학습 당시 로컬 경로가 남아 있어 정정
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    shutil.copy(w_p, out / "diffusion_pytorch_model.safetensors")   # trainer.pt(optimizer) 제외

    figs = _stage_figures(out, Path(args.figures_dir) if args.figures_dir else None, {
        "recon_ko": "recon_ko.png", "recon_en": "recon_en.png",
        "gen_ko": "gencmp_ko_release.png", "gen_en": "gencmp_en_release.png"})
    sec = []
    if "recon_ko" in figs or "recon_en" in figs:
        sec.append("\n### Before / after\n\nSame input, roundtrip through each VAE "
                   "(source / original / this model), with per-line MSE and SSIM.\n\n")
        if "recon_ko" in figs:
            sec.append(f"Korean — the original smears dense syllables, this model keeps the strokes:\n\n"
                       f"![Korean reconstruction]({_url(args.repo, figs['recon_ko'])})\n\n")
        if "recon_en" in figs:
            sec.append(f"English — the recipe is script-neutral, so Latin improves as well:\n\n"
                       f"![English reconstruction]({_url(args.repo, figs['recon_en'])})\n\n")
    if "gen_ko" in figs or "gen_en" in figs:
        sec.append(f"\n### What it changes in generation\n\nThe VAE only sets the ceiling — the T5 "
                   f"decoder must be re-adapted to reach it (its latent space moved). Original English "
                   f"pretrained Eruku with the original VAE vs. the paired release "
                   f"[`{args.model_repo}`](https://huggingface.co/{args.model_repo}), which uses this VAE. "
                   f"Columns: style reference | target rendered in that font | before | after.\n\n")
        if "gen_ko" in figs:
            sec.append("Korean, with Korean-font style references — note the baseline has seen neither "
                       "Hangul glyphs nor Korean-font styles:\n\n"
                       f"![Korean generation]({_url(args.repo, figs['gen_ko'])})\n\n")
        if "gen_en" in figs:
            sec.append("English, with Latin style references that are in distribution for both models, "
                       "so only the model differs — English survives, and strokes come out crisper:\n\n"
                       f"![English generation]({_url(args.repo, figs['gen_en'])})\n\n")
    card = VAE_CARD.replace("{FIGURES}", "".join(sec) if sec else "")
    for k, v in {"VAE_REPO": args.repo, "MODEL_REPO": args.model_repo,
                 "ORIG_KO_MSE": args.orig_ko, "ORIG_KO_SSIM": args.orig_ko_ssim,
                 "ORIG_HARD_MSE": args.orig_hard, "ORIG_EN_MSE": args.orig_en,
                 "NEW_KO_MSE": args.new_ko, "NEW_KO_SSIM": args.new_ko_ssim,
                 "NEW_HARD_MSE": args.new_hard, "NEW_EN_MSE": args.new_en}.items():
        card = card.replace("{" + k + "}", str(v))
    (out / "README.md").write_text(card, encoding="utf-8")

    print(f"[vae] 스테이징 완료 {out}")
    for p in sorted(out.iterdir()):
        print(f"       {p.name}  {p.stat().st_size/1e6:.1f}MB")
    if args.push:
        _push(_api(args.token), out, args.repo,
              f"Korean-adapted VAE (full finetune, {src.name})")
    else:
        print("[vae] dry-run — 업로드는 --push")


# ─────────────────────────────────────────────────────────────────────────────
# model 서브커맨드
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CARD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: image-to-image
language:
- ko
- en
tags:
- eruku
- handwriting-generation
- styled-text-generation
- korean
- autoregressive
- vision
base_model: blowing-up-groundhogs/eruku
base_model_relation: finetune
---

# eruku_korean

Korean styled handwriting line-image generation, fine-tuned from
[`blowing-up-groundhogs/eruku`](https://huggingface.co/blowing-up-groundhogs/eruku)
(autoregressive styled handwriting generation, [arXiv 2510.23240](https://arxiv.org/abs/2510.23240)).
Give it a style reference line and a Korean/English string; it draws that string in that style.

Training code and experiment log: <https://github.com/HERIUN/Eruku_korean_finetuning>

> **This revision requires the paired Korean VAE.** `config.vae_name_or_path` points at
> [`{VAE_REPO}`](https://huggingface.co/{VAE_REPO}). The VAE was fine-tuned with
> encoder+decoder unfrozen, so its latent space moved and the T5 decoder here was
> re-adapted on top of it. Swapping in the original `emuru_vae` produces garbage.

## Quick start

```python
from transformers import AutoModel
from PIL import Image

model = AutoModel.from_pretrained("{MODEL_REPO}", trust_remote_code=True).eval().cuda()

img = model.generate_handwriting(
    style_image=Image.open("style_line.png"),   # one line of handwriting
    style_text="조용히 책을 읽었다.",              # transcription of the style line (optional, but helps a lot)
    gen_text="한국어 손글씨 생성",                 # what to write
    cfg_scale=1.5,
)
img.save("out.png")
```

- `style_text` — passing the style line's transcription markedly improves fidelity.
- `cfg_scale` — 1.0 is the most faithful on the eval battery; 1.5-2.0 pushes style strength.
- `bbox_crop=True` (default) crops the style image to its ink bounding box. Real scans carry
  page margins that blow up the VAE latent std and make generation run away or come back
  blank; this closes that gap. Pass `bbox_crop=False` for crops that already fill the frame.

## Results

CER measured with a Korean HTR reader (floor on font renders ~0.00; easyocr saturates at
0.236 even on Korean ground truth and cannot be used). n=100 sentences split into length
buckets, cfg=1.0, coherent sentences, seen fonts.

| bucket | decoder-only VAE | full VAE, before re-adaptation | **this model** |
|---|---|---|---|
| Korean, all | 0.203 | 0.211 | **{KO_ALL}** |
| Korean, short (1-3 words) | 0.497 | 0.540 | **{KO_SHORT}** |
| Korean, mid (4-8 words) | — | — | {KO_MID} |
| Korean, long (9-15 words) | — | — | {KO_LONG} |
| English, all | 0.084 | 0.056 | **{EN_ALL}** |

Two things had to happen together: the Korean VAE raised the reconstruction ceiling
(roundtrip MSE -83%), and a short-text-weighted re-adaptation of T5 closed the gap to that
ceiling — short Korean strings were the actual bottleneck (CER 0.54 -> {KO_SHORT}).
{FIGURES}

## How it was trained

English pretrained -> Korean (`--english-frac 0.15` to hold English) -> VAE swap
(`--reset-vae`) + re-adaptation with short-text emphasis, {STEPS} optimizer steps total.
Data is synthesized online: fonts act as writers (83 Korean handwriting/display + 177 Latin),
text is sampled from a Korean corpus, rendered and augmented per sample, so every sample is
unique. Backbone: T5-large + ByT5 byte tokenizer (byte-level input handles Hangul without a
Korean vocab), frozen VAE.

## Limitations

- **Short strings are still the weak spot** (CER {KO_SHORT} vs {KO_LONG} for long lines) —
  fewer latent columns means less context for the autoregressive decoder.
- Unseen extreme display/brush fonts degrade; the training set has 99 fonts, upstream had ~100k.
- The VAE snaps ink density to crisp black, so it does not preserve faint/pencil strokes
  (prototype collapse; see `docs/VAE_ROBUSTNESS.md` in the repo).
- Long **English** lines are weaker than the original English model — Korean is the target here.

License: Apache-2.0, inherited from the base model.
"""


def _staging_code(out: Path, code_from: str):
    """기존 repo 의 remote code 를 받아 bbox 패치 후 스테이징. (config.json 원본도 반환)"""
    from huggingface_hub import hf_hub_download
    out.mkdir(parents=True, exist_ok=True)
    cfg_json = json.loads(Path(hf_hub_download(code_from, "config.json")).read_text())
    conf_src = Path(hf_hub_download(code_from, "configuration_eruku.py")).read_text()
    mod_src = Path(hf_hub_download(code_from, "modeling_eruku.py")).read_text()
    (out / "configuration_eruku.py").write_text(conf_src, encoding="utf-8")
    (out / "modeling_eruku.py").write_text(patch_modeling(mod_src), encoding="utf-8")
    print(f"[model] remote code 스테이징 + bbox 패치 적용 ({code_from})")
    return cfg_json


def _load_release_class(staging: Path):
    """스테이징한 modeling_eruku.py 에서 릴리즈 클래스/Config 를 로드(패치본 그대로 검증)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("modeling_eruku.ErukuForConditionalGeneration", str(staging))
    conf = get_class_from_dynamic_module("configuration_eruku.ErukuConfig", str(staging))
    return cls, conf


def _ckpt_state(ckpt: Path):
    import torch
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = ck["model"] if "model" in ck else ck
    step = ck.get("step", "?")
    if any(k.startswith("module.") for k in st):
        st = {k[len("module."):]: v for k, v in st.items()}
    n0 = len(st)
    # t5_to_ocr: 로컬 학습 클래스에만 있는 OCR head (alpha=1.0 이라 loss 미사용). 릴리즈 클래스엔 없음
    st = {k: v for k, v in st.items() if not k.startswith("t5_to_ocr.")}
    print(f"[model] ckpt step={step}, {n0}키 → t5_to_ocr {n0-len(st)}키 strip → {len(st)}키 "
          f"(vae.* {sum(k.startswith('vae.') for k in st)}키 포함)")
    return st, step


def cmd_model(args):
    import torch
    ckpt, out = Path(args.ckpt), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    cfg_json = _staging_code(out, args.code_from)
    cls, conf_cls = _load_release_class(out)

    # 빌드 시점엔 로컬 VAE 경로로 인스턴스화(네트워크/미발행 repo 의존 제거),
    # 저장 직전에 config 를 업로드용 repo id 로 바꾼다.
    cfg = conf_cls(**{k: v for k, v in cfg_json.items()
                      if k in ("t5_name_or_path", "tokenizer_name_or_path", "slices_per_query",
                               "channels", "vae_latent_dim", "cfg_scale")})
    cfg.vae_name_or_path = str(args.local_vae)
    model = cls(cfg)

    st, step = _ckpt_state(ckpt)
    missing, unexpected = model.load_state_dict(st, strict=False)
    # 조용한 부분 로드 = 잘못된 가중치를 발행하는 최악의 실패 → 하나라도 있으면 중단
    print(f"[model] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"        missing={list(missing)[:8]}\n        unexpected={list(unexpected)[:8]}")
        raise SystemExit("[model] 키 불일치 → abort (릴리즈 클래스와 학습 클래스 구조 확인 필요)")

    if args.verify_cer:
        _verify_cer(model, args)

    cfg.vae_name_or_path = args.vae_repo          # 발행용: 페어 VAE repo
    model.config.vae_name_or_path = args.vae_repo
    model.save_pretrained(str(out), safe_serialization=not args.bin)

    # auto_map(trust_remote_code 진입점) 보존. save_pretrained 는 AutoModel 항목만 다시 써서
    # AutoConfig 매핑을 떨어뜨린다 → 그러면 AutoConfig/AutoModel.from_pretrained 가 커스텀
    # config 클래스를 못 찾아 로드가 실패한다. 원본 항목을 채워 넣는다(기존 값은 유지).
    saved = json.loads((out / "config.json").read_text())
    am = dict(cfg_json.get("auto_map", {}))
    am.update(saved.get("auto_map", {}))
    if am != saved.get("auto_map"):
        saved["auto_map"] = am
        (out / "config.json").write_text(json.dumps(saved, indent=2) + "\n")
        print(f"[model] config.json auto_map 보완 → {sorted(am)}")
    assert saved.get("vae_name_or_path") == args.vae_repo, saved.get("vae_name_or_path")

    figs = _stage_figures(out, Path(args.figures_dir) if args.figures_dir else None, {
        "gen_ko": "gencmp_ko_release.png", "gen_en": "gencmp_en_release.png"})
    sec = []
    if figs:
        sec.append("\n### Before / after\n\nSame target text through the original English pretrained "
                   "Eruku (with the original VAE) and through this release. Columns: style reference | "
                   "target rendered in that font | before | after.\n\n")
        if "gen_ko" in figs:
            sec.append("Korean, with Korean-font style references. The baseline has seen neither Hangul "
                       "glyphs nor Korean-font styles, so it returns nothing usable:\n\n"
                       f"![Korean generation]({_url(args.repo, figs['gen_ko'])})\n\n")
        if "gen_en" in figs:
            sec.append("English, with Latin style references that are in distribution for both models — "
                       "English is kept, not traded away for Korean, and strokes come out crisper "
                       "(the Korean VAE is a better decoder for Latin too):\n\n"
                       f"![English generation]({_url(args.repo, figs['gen_en'])})\n\n")

    card = MODEL_CARD.replace("{FIGURES}", "".join(sec) if sec else "")
    for k, v in {"MODEL_REPO": args.repo, "VAE_REPO": args.vae_repo, "STEPS": str(step),
                 "KO_ALL": args.ko_all, "KO_SHORT": args.ko_short, "KO_MID": args.ko_mid,
                 "KO_LONG": args.ko_long, "EN_ALL": args.en_all}.items():
        card = card.replace("{" + k + "}", str(v))
    (out / "README.md").write_text(card, encoding="utf-8")

    print(f"[model] 스테이징 완료 {out}")
    for p in sorted(out.iterdir()):
        print(f"       {p.name}  {p.stat().st_size/1e6:.1f}MB")
    if args.push:
        # 구 revision 의 pytorch_model.bin 이 남으면 weight 파일 2개가 공존 → 삭제
        dele = None if args.bin else ["pytorch_model.bin"]
        _push(_api(args.token), out, args.repo,
              f"Update to full-VAE combo (T5 step {step} + Korean VAE); add ink-bbox style crop",
              delete_patterns=dele)
    else:
        print("[model] dry-run — 업로드는 --push")


def _verify_cer(hf_model, args):
    """변환본을 **공개 API 그대로**(generate_handwriting) 돌려 한글 CER 을 잰다.

    가중치 키 정합성(missing/unexpected 0)만으로는 "릴리즈 경로로 실제 잘 생성되는가"를
    보장하지 못한다(릴리즈 클래스와 학습 클래스는 style prefix 처리·crop 규약이 다르다).
    그래서 로컬 harness 수치(eval_htr_cer.py)와 같은 리더로 재서 임계 이내인지 확인한다.
    추가로 실제 스캔 손글씨 한 장에 bbox 크롭 on/off 를 돌려 패치 동작을 확인한다.
    """
    import random
    import numpy as np
    import torch
    from PIL import Image
    from infer_show import render_in_font
    from eval_echo_metrics import build_texts, font_can_render
    from eval_htr_cer import to_htr_input, htr_read, norm
    from train_aux_htr import load_pretrained_htr, cer as char_cer

    dev = torch.device(args.device)
    model = hf_model.to(dev).eval()
    htr = load_pretrained_htr(Path(args.htr_checkpoint)).to(dev).eval()
    rng = random.Random(0)
    texts = build_texts(args.n_cer, rng, english=False, coherent=True, max_words=15)
    fonts = sorted((HERE / "assets/fonts_korean_v2/train").glob("*.ttf"))

    rows = []
    for text in texts:
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text)), None)
        if fp is None:
            continue
        gt = render_in_font(fp, text)
        try:
            # 폰트 렌더는 이미 잉크에 타이트 → bbox 크롭 off (실제 스캔에서만 필요)
            out = model.generate_handwriting(style_image=Image.fromarray(gt), gen_text=text,
                                            style_text=text, cfg_scale=1.0,
                                            max_new_tokens=args.max_new, bbox_crop=False)
        except Exception as e:
            print(f"  [skip] {e}")
            continue
        cg = char_cer([norm(htr_read(htr, to_htr_input(np.array(out), dev), dev)[0])], [norm(text)])
        cgt = char_cer([norm(htr_read(htr, to_htr_input(gt, dev), dev)[0])], [norm(text)])
        rows.append((len(text.split()), cg, cgt))
        print(f"  [cer] {len(rows):3d}/{args.n_cer} nw={len(text.split()):2d} "
              f"gen={cg:.3f} gt={cgt:.3f}", flush=True)

    if not rows:
        raise SystemExit("[cer] 샘플을 하나도 생성하지 못함 → abort")
    a = np.array([[r[1], r[2]] for r in rows])
    short = np.array([r[1] for r in rows if r[0] <= 3])
    print(f"[cer] n={len(rows)} 한글 gen CER={a[:, 0].mean():.3f} (GT floor {a[:, 1].mean():.3f})"
          + (f" | 단문(1~3) n={len(short)} {short.mean():.3f}" if len(short) else ""))
    if a[:, 0].mean() > args.cer_max:
        raise SystemExit(f"[cer] {a[:, 0].mean():.3f} > 임계 {args.cer_max} → abort (변환/조합 확인)")

    # bbox 패치 smoke test: 실제 스캔 손글씨 style (여백 있음) 에서 runaway/blank 안 나는지
    hw = sorted((HERE / "custom_datasets/hw_line_sample").glob("line_*.png"))
    if hw:
        labels = {}
        lab_p = HERE / "custom_datasets/hw_line_sample/label.txt"
        if lab_p.exists():
            for ln in lab_p.read_text(encoding="utf-8").splitlines():
                if "\t" in ln:
                    k, v = ln.split("\t", 1)
                    labels[k] = v
        p = hw[0]
        stext = labels.get(p.name, "")
        gen_text = "한국어 손글씨 생성"
        for flag in (True, False):
            out = model.generate_handwriting(style_image=Image.open(p), gen_text=gen_text,
                                             style_text=stext, cfg_scale=1.5,
                                             max_new_tokens=args.max_new, bbox_crop=flag)
            read = htr_read(htr, to_htr_input(np.array(out), dev), dev)[0]
            c = char_cer([norm(read)], [norm(gen_text)])
            print(f"[bbox] {p.name} bbox_crop={str(flag):5s} → w={out.size[0]:4d} CER={c:.3f} 읽힘='{read}'")
    model.to("cpu")
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", default=None, help="HF 토큰(기본: HF_TOKEN env 또는 hf auth login 캐시)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("vae", help="한글 적응 VAE 발행 (먼저 실행)")
    v.add_argument("--src", default=str(DEF_VAE_SRC))
    v.add_argument("--repo", default=DEF_VAE_REPO)
    v.add_argument("--model-repo", default=DEF_MODEL_REPO)
    v.add_argument("--out", default="/tmp/hf_emuru_vae_korean")
    v.add_argument("--push", action="store_true")
    v.add_argument("--figures-dir", default=None,
                   help="카드에 넣을 figure 디렉토리. recon_{ko,en}.png(_eval_vae_recon.py --label-lang en) + "
                        "gencmp_{ko,en}_release.png(_eval_gen_compare.py --cats ko_release en_release). "
                        "있는 파일만 samples/ 로 올리고 카드에 삽입")
    # 모델카드 수치 (_eval_vae_recon.py 고정 probe 평균)
    v.add_argument("--orig-ko", default="0.0188"); v.add_argument("--orig-ko-ssim", default="0.904")
    v.add_argument("--orig-hard", default="0.0197"); v.add_argument("--orig-en", default="0.0020")
    v.add_argument("--new-ko", default="0.0031"); v.add_argument("--new-ko-ssim", default="0.977")
    v.add_argument("--new-hard", default="0.0128"); v.add_argument("--new-en", default="0.0012")
    v.set_defaults(func=cmd_vae)

    m = sub.add_parser("model", help="상위 Eruku 모델 발행")
    m.add_argument("--ckpt", default=str(DEF_CKPT))
    m.add_argument("--repo", default=DEF_MODEL_REPO)
    m.add_argument("--vae-repo", default=DEF_VAE_REPO, help="config.vae_name_or_path 로 박히는 페어 VAE")
    m.add_argument("--local-vae", default=str(DEF_VAE_SRC), help="빌드/검증 시 쓸 로컬 VAE 경로")
    m.add_argument("--code-from", default=DEF_MODEL_REPO, help="remote code(modeling/configuration) 출처 repo")
    m.add_argument("--out", default="/tmp/hf_eruku_korean")
    m.add_argument("--push", action="store_true")
    m.add_argument("--figures-dir", default=None,
                   help="카드에 넣을 생성 비교 figure 디렉토리(gencmp_{ko,en}_release.png). "
                        "_eval_gen_compare.py --cats ko_release en_release 로 생성")
    m.add_argument("--bin", action="store_true", help="safetensors 대신 pytorch_model.bin 으로 저장")
    m.add_argument("--verify-cer", action="store_true",
                   help="변환본을 공개 API 로 돌려 한글 CER 측정 + bbox 크롭 smoke test (GPU 필요)")
    m.add_argument("--htr-checkpoint", default="finetune_runs/aux_htr_ko/htr_s20000")
    m.add_argument("--n-cer", type=int, default=24, help="--verify-cer 샘플 수")
    m.add_argument("--cer-max", type=float, default=0.20, help="이 CER 을 넘으면 abort")
    m.add_argument("--device", default="cuda")
    m.add_argument("--max-new", type=int, default=480)
    # 모델카드 수치 (eval_htr_cer.py, n=100, cfg=1.0)
    m.add_argument("--ko-all", default="0.127"); m.add_argument("--ko-short", default="0.286")
    m.add_argument("--ko-mid", default="0.049"); m.add_argument("--ko-long", default="0.059")
    m.add_argument("--en-all", default="0.045")
    m.set_defaults(func=cmd_model)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
