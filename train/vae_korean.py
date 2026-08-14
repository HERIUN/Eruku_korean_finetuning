"""VAE 한글 적응 finetune (원본 emuru VAE 레시피 재현, 한글 HTR 가독성 신호 사용).

배경: Eruku 는 diffusers AutoencoderKL(`emuru_vae`)을 동결로 쓰고 모든 조건/타깃이 이 VAE latent.
VAE 는 `in=3ch(styled RGB) → out=1ch(clean B/W)` 디노이징 AE + latent_channels=1·downsample8 병목이라
한글 재구성 오차가 영어의 ~10배 = 한글 fidelity 하드 상한. 원본 학습의 가독성 핵심은 L1 이 아니라
**HTR(OCR) loss**(복원을 OCR 이 읽어 글자가 맞게 나오게 강제)인데, 그 HTR 이 라틴 전용이라 한글은
최적화된 적이 없었다. → train/aux_htr.py 로 만든 **한글 HTR** 을 붙여 VAE 를 한글에 적응.

loss = L1(bw) + kl_weight·KL(full 만) + htr_weight·HTR항
  HTR항 mode:
    ce   : 한글 HTR 분류 loss(SmoothCE, AR readability). 복원을 HTR 이 읽어 글자 맞추게.  [원본]
    feat : HTR feature-유사도(복원/clean 을 htr.feature_extractor 통과, feature L1). head 불필요, 안정.
    both : ce + feat.

--train-part:
  decoder(1차): encoder+quant_conv 동결, decoder+post_quant_conv 만 → latent 공간 불변 → 기존 Eruku 재학습 불필요.
  full        : enc+dec 모두 → 상한↑ 이지만 latent 이동 → Eruku 재적응 필요.

결과 `save_pretrained(out/vae_sXXXX)` → Eruku `--vae-checkpoint` 로 로드(교체 검증: infer.show/eval.cer 의 --vae-checkpoint).

예:
  python train/vae_korean.py --train-part decoder --htr-checkpoint finetune_runs/aux_htr_ko/htr_s20000 \
      --out finetune_runs/vae_korean_dec --batch-size 8 --grad-accum 4 --max-steps 15000 --val-every 500
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))

from diffusers import AutoencoderKL

from models.smooth_ce import SmoothCrossEntropyLoss
from models.teacher_forcing import NoisyTeacherForcing
from custom_datasets.korean_alphabet import get_korean_alphabet
from korean_aux_dataset import KoreanAuxDataset, aux_collate
from train.aux_htr import load_pretrained_htr


from infer.show import render_in_font
from korean_aux_dataset import _fit_width, H as _AUXH

# 영어 회귀 모니터용 고정 probe (held-out 폰트로 라틴 렌더). full 학습 중 영어 recon 유지 확인.
_TEST_FONTS = HERE / "assets" / "fonts_korean_v2" / "test"
_EN_PROBE_TEXTS = [
    "Hello World", "evaluate model performance", "The quick brown fox jumps over the lazy dog",
    "Autoregressive text image generation 2024", "colorful flowers bloom in spring",
    "held out font generalization check",
]
_PROBE_FONTS = ["DoHyeon.ttf", "GowunBatang.ttf", "Gaegu.ttf"]


# ── eval: held-out 폰트 clean-bw roundtrip MSE/SSIM (docs/_vae_capacity 와 동일 지표) ──
@torch.no_grad()
def build_val_bw(n, seed, fonts_dir):
    ds = KoreanAuxDataset(length=n, seed=seed, fonts_dir=fonts_dir)
    return torch.stack([ds[i]["bw"] for i in range(n)], 0)  # [n,1,64,768] [-1,1] clean


@torch.no_grad()
def build_en_probe():
    """영어 회귀 모니터: 고정 영어 텍스트×held-out 폰트 → clean bw [N,1,64,768] [-1,1]."""
    out = []
    for fn in _PROBE_FONTS:
        fp = _TEST_FONTS / fn
        if not fp.exists():
            continue
        for t in _EN_PROBE_TEXTS:
            arr = render_in_font(fp, t)                          # [h,w] uint8 ink~0 bg~255
            x = torch.from_numpy(arr.astype(np.float32) / 255.0)[None, None]
            w64 = max(1, int(round(_AUXH * arr.shape[1] / arr.shape[0])))
            x = torch.nn.functional.interpolate(x, size=(_AUXH, w64), mode="bilinear", align_corners=False)
            out.append(_fit_width((x.clamp(0, 1) * 2 - 1)[0]))   # [1,64,768]
    return torch.stack(out, 0)


@torch.no_grad()
def evaluate_vae(vae, val_bw, device, bs=16):
    was = vae.training; vae.eval()
    mses, ssims = [], []
    for k in range(0, val_bw.size(0), bs):
        x = val_bw[k:k + bs].to(device)
        x3 = x.repeat(1, 3, 1, 1)                       # VAE 입력=3ch (grayscale 복제, _vae_capacity 와 동일)
        z = vae.encode(x3).latent_dist.mode()
        rec = vae.decode(z).sample.clamp(-1, 1)         # [b,1,64,768]
        xg = ((x[:, 0] + 1) / 2).cpu().numpy()
        rg = ((rec[:, 0] + 1) / 2).cpu().numpy()
        for a, b in zip(xg, rg):
            mses.append(float(np.mean((a - b) ** 2)))
            ssims.append(float(ssim(a, b, data_range=1.0)))
    if was:
        vae.train()
    return float(np.mean(mses)), float(np.mean(ssims))


@torch.no_grad()
def dump_recon(vae, val_bw, device, path, n=8):
    was = vae.training; vae.eval()
    x = val_bw[:n].to(device)
    rec = vae.decode(vae.encode(x.repeat(1, 3, 1, 1)).latent_dist.mode()).sample.clamp(-1, 1)
    xg = ((x[:, 0] + 1) / 2 * 255).byte().cpu().numpy()
    rg = ((rec[:, 0] + 1) / 2 * 255).byte().cpu().numpy()
    rows = []
    for a, b in zip(xg, rg):
        rows.append(np.vstack([a, np.full((3, a.shape[1]), 128, np.uint8), b,
                               np.full((6, a.shape[1]), 200, np.uint8)]))
    Image.fromarray(np.vstack(rows)).save(path)
    if was:
        vae.train()


def set_trainable(vae, train_part):
    if train_part == "full":
        vae.requires_grad_(True); vae.train()
        return True
    vae.requires_grad_(False)
    for m in (vae.decoder, vae.post_quant_conv):
        m.requires_grad_(True); m.train()
    vae.encoder.eval(); vae.quant_conv.eval()
    return False


# ── Laplacian 피라미드 loss (방법 c: 주파수 대역별 재가중) ──────────────
# rec/tgt 를 대역별(band-pass)로 분해해 고주파(얇은 획·경계) 대역에 큰 가중.
# rec==tgt 면 모든 대역 일치 → loss 0 (flat L1 과 전역최소 동일, on-manifold).
def _gauss_kernel(device, dtype):
    k = torch.tensor([1., 4., 6., 4., 1.], device=device, dtype=dtype)
    k2 = torch.outer(k, k)
    return (k2 / k2.sum()).view(1, 1, 5, 5)


def _blur(x, kernel):
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel)


def _pyr_down(x, kernel):
    return _blur(x, kernel)[:, :, ::2, ::2]


def _pyr_up(x, kernel, out_hw):
    up = F.interpolate(x, size=out_hw, mode="nearest")
    return _blur(up, kernel)


def laplacian_pyramid_loss(rec, tgt, kernel, levels, alphas):
    """levels 개 band-pass + 1 개 저주파 잔차. alphas 길이 = levels+1 (앞=고주파)."""
    loss = rec.new_zeros(())
    cr, ct = rec, tgt
    for l in range(levels):
        dr, dt = _pyr_down(cr, kernel), _pyr_down(ct, kernel)
        lap_r = cr - _pyr_up(dr, kernel, cr.shape[-2:])   # 스케일 l 의 band-pass
        lap_t = ct - _pyr_up(dt, kernel, ct.shape[-2:])
        loss = loss + alphas[l] * (lap_r - lap_t).abs().mean()
        cr, ct = dr, dt
    loss = loss + alphas[levels] * (cr - ct).abs().mean()   # 저주파 잔차
    return loss


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="finetune_runs/vae_korean_dec")
    p.add_argument("--vae-checkpoint", default="blowing-up-groundhogs/emuru_vae")
    p.add_argument("--htr-checkpoint", required=True, help="train/aux_htr.py 결과(htr_sXXXX)")
    p.add_argument("--train-part", choices=["decoder", "full"], default="decoder")
    p.add_argument("--htr-loss-mode", choices=["ce", "feat", "both"], default="ce")
    p.add_argument("--htr-weight", type=float, default=0.3)
    p.add_argument("--kl-weight", type=float, default=1e-6, help="full 일 때만 적용")
    p.add_argument("--logvar-weighting", action="store_true",
                   help="원본 AutoencoderLoss 의 학습형 uncertainty weighting 복원: "
                        "nll(=l1+htr) / exp(log_var) + log_var (KL 은 밖). log_var 는 학습 파라미터.")
    p.add_argument("--logvar-init", type=float, default=0.0, help="log_var 초기값(원본 0.0)")
    p.add_argument("--pyramid-weight", type=float, default=0.0,
                   help="방법(c) Laplacian 피라미드 loss 가중(0=off). base_L1 + w·L_pyr.")
    p.add_argument("--pyramid-levels", type=int, default=3, help="band-pass 레벨 수(64px→3=64/32/16 + 잔차)")
    p.add_argument("--pyramid-alphas", type=float, nargs="+", default=None,
                   help="레벨별 가중(길이=levels+1, 앞=고주파). 미지정 시 [1,.5,.25,.125...] 자동")
    p.add_argument("--noise-prob", type=float, default=0.3, help="ce: NoisyTeacherForcing 확률(원본 0.3)")
    p.add_argument("--writer-weight", type=float, default=0.0, help="WID(현재 0=off)")
    p.add_argument("--max-steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=8, help="이미지 수(rgb→bw 페어)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=500, help="0=off")
    p.add_argument("--val-samples", type=int, default=64)
    p.add_argument("--train-fonts-dir", default=None, help="학습 폰트 dir(기본 fonts_korean_v2/train). 확대셋 지정용")
    p.add_argument("--augment", action="store_true", help="shape 증강(morph+회전+elastic warp) = 스타일 일반화")
    p.add_argument("--uniform-freq", action="store_true", help="논문식 균일 음절빈도(모든 글리프 균등 노출)")
    p.add_argument("--val-fonts-dir", default=None,
                   help="val 한글 폰트 dir. 기본(None)=--val-font-frac 으로 학습 폰트에서 분할. "
                        "held-out 은 assets/fonts_korean_v2/test (최종 평가 전용 권장)")
    p.add_argument("--val-font-frac", type=float, default=0.1,
                   help="--val-fonts-dir 미지정 시 학습 폰트에서 val 로 떼는 비율(학습 제외, seed 로 결정적). "
                        "0=분할 안 함")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None, help="vae_sXXXX 디렉토리(가중치+trainer.pt)")
    return p


def main():
    args = make_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)

    # ── VAE ──
    src = args.resume if args.resume else args.vae_checkpoint
    print(f"loading VAE: {src}")
    vae = AutoencoderKL.from_pretrained(src).to(device)
    train_encoder = set_trainable(vae, args.train_part)
    nt = sum(p.numel() for p in vae.parameters() if p.requires_grad) / 1e6
    print(f"train-part={args.train_part}  trainable={nt:.1f}M  train_encoder={train_encoder}")

    # ── 한글 HTR (동결) ──
    alpha = get_korean_alphabet(); A = len(alpha)
    htr = load_pretrained_htr(Path(args.htr_checkpoint)).to(device)
    assert htr.fc.out_features == A, f"HTR alphabet_size({htr.fc.out_features}) != korean({A}) — 알파벳 불일치"
    htr.requires_grad_(False); htr.eval()
    smooth_ce = SmoothCrossEntropyLoss(tgt_pad_idx=alpha.pad)
    noisy = NoisyTeacherForcing(A, alpha.num_extra_tokens, args.noise_prob)
    print(f"HTR loaded (mode={args.htr_loss_mode}, weight={args.htr_weight})")

    # ── 학습형 loss weighting(원본 AutoencoderLoss 재현): nll/exp(log_var)+log_var ──
    log_var = None
    if args.logvar_weighting:
        log_var = torch.nn.Parameter(torch.ones((), device=device) * args.logvar_init)
        print(f"logvar-weighting ON (init={args.logvar_init})")

    # ── 피라미드 loss(방법 c) 설정 ──
    pyr_kernel = pyr_alphas = None
    if args.pyramid_weight > 0:
        pyr_kernel = _gauss_kernel(device, torch.float32)
        if args.pyramid_alphas is not None:
            assert len(args.pyramid_alphas) == args.pyramid_levels + 1, \
                f"pyramid-alphas 길이({len(args.pyramid_alphas)}) != levels+1({args.pyramid_levels + 1})"
            pyr_alphas = args.pyramid_alphas
        else:
            pyr_alphas = [0.5 ** i for i in range(args.pyramid_levels + 1)]  # 고주파 우선 [1,.5,.25,...]
        print(f"pyramid loss ON (w={args.pyramid_weight}, levels={args.pyramid_levels}, alphas={pyr_alphas})")

    opt_params = [p for p in vae.parameters() if p.requires_grad]
    if log_var is not None:
        opt_params = opt_params + [log_var]
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-2)
    start_step = 0
    if args.resume:
        tp = Path(args.resume) / "trainer.pt"
        if tp.exists():
            st = torch.load(tp, map_location="cpu")
            try:
                optimizer.load_state_dict(st["optimizer"])
            except Exception as e:
                print(f"  optim restore 실패(무시): {e}")
            for g in optimizer.param_groups:
                g["lr"] = args.lr
            if log_var is not None and "log_var" in st:
                with torch.no_grad():
                    log_var.copy_(torch.as_tensor(st["log_var"], device=device))
                print(f"  resume log_var={float(log_var.detach()):.4f}")
            start_step = int(st.get("step", 0))
            print(f"  resume step={start_step}")

    (out / "vae_train_config.txt").open("a", encoding="utf-8").write(f"{vars(args)}\n")

    # 폰트 단위 train/val 분할: val 폰트는 학습에서 제외(--val-fonts-dir 지정 시엔 분할 안 함)
    train_fonts = val_fonts = None
    if not args.val_fonts_dir and args.val_font_frac > 0:
        from korean_split_dataset import split_train_fonts
        train_fonts, val_fonts = split_train_fonts(args.val_font_frac, seed=args.seed,
                                                   fonts_dir=args.train_fonts_dir)
        print(f"폰트 분할: train {len(train_fonts)} / val {len(val_fonts)} "
              f"(val-font-frac={args.val_font_frac}, seed={args.seed})")

    total = args.max_steps * args.grad_accum * args.batch_size
    ds = KoreanAuxDataset(length=total, seed=args.seed + start_step,
                          fonts_dir=(train_fonts or args.train_fonts_dir), augment=args.augment,
                          uniform_freq=args.uniform_freq)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=aux_collate, drop_last=True, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    val_bw = val_en = None
    if args.val_every > 0:
        if args.val_fonts_dir and Path(args.val_fonts_dir).is_dir():
            vfd = args.val_fonts_dir
        elif val_fonts:
            vfd = val_fonts   # 학습에서 제외한 분할 폰트 → 본 적 없는 폰트로 val
        else:
            vfd = None
            print("  [val] 폰트 분할 없음(--val-font-frac 0) → 학습 폰트에서 텍스트만 분리")
        val_bw = build_val_bw(args.val_samples, args.seed + 99999, vfd)   # held-out 한글(+mixed)
        val_en = build_en_probe()                                        # 영어 회귀 모니터
        m0, s0 = evaluate_vae(vae, val_bw, device)
        me0, se0 = evaluate_vae(vae, val_en, device)
        print(f"[val@baseline] KO MSE={m0:.5f} SSIM={s0:.4f} | EN MSE={me0:.5f} SSIM={se0:.4f}")
        dump_recon(vae, val_bw, device, out / "samples" / "recon_baseline.png")

    tl = (out / "vae_train.csv").open("a", newline="", encoding="utf-8")
    tw = csv.writer(tl)
    if tl.tell() == 0:
        tw.writerow(["step", "loss", "l1", "htr", "pyr", "kl", "sec_per_step"])
    vl = (out / "vae_val.csv").open("a", newline="", encoding="utf-8")
    vw = csv.writer(vl)
    if vl.tell() == 0:
        vw.writerow(["step", "val_mse_ko", "val_ssim_ko", "val_mse_en", "val_ssim_en"])

    def save(step):
        d = out / f"vae_s{step}"
        vae.save_pretrained(d)
        ckpt = {"step": step, "optimizer": optimizer.state_dict()}
        if log_var is not None:
            ckpt["log_var"] = float(log_var.detach())
        torch.save(ckpt, d / "trainer.pt")
        print(f"  saved {d}")

    def htr_loss(rec, bw, text, tlen, tmask, pad):
        """rec/bw: [B,1,64,768] [-1,1]. mode 에 따라 ce/feat/both."""
        ce = feat = torch.zeros((), device=device)
        if args.htr_loss_mode in ("ce", "both"):
            noisy_text = noisy(text, tlen)
            o = htr(rec, noisy_text[:, :-1], tmask, pad[:, :-1])   # HTR 이 복원을 읽음
            ce = smooth_ce(o, text[:, 1:])
        if args.htr_loss_mode in ("feat", "both"):
            fr = htr.feature_extractor(rec)
            with torch.no_grad():
                ft = htr.feature_extractor(bw)                     # clean 글리프의 OCR feature
            feat = (fr - ft).abs().mean()
        return ce + feat, ce, feat

    print(f"training {start_step} -> {args.max_steps}  (virtual batch = {args.batch_size}×{args.grad_accum})")
    step, micro = start_step, 0
    optimizer.zero_grad(set_to_none=True)
    run = {"loss": 0.0, "l1": 0.0, "htr": 0.0, "pyr": 0.0, "kl": 0.0, "n": 0}
    t0 = time.time()
    it = iter(loader)
    while step < args.max_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        rgb = b["rgb"].to(device, non_blocking=True)
        bw = b["bw"].to(device, non_blocking=True)
        text = b["text_logits_s2s"].to(device); tlen = b["texts_len"].to(device)
        tmask = b["tgt_key_mask"].to(device); pad = b["tgt_key_padding_mask"].to(device)

        if train_encoder:
            posterior = vae.encode(rgb).latent_dist
            z = posterior.sample(); kl = posterior.kl().mean()
        else:
            with torch.no_grad():
                z = vae.encode(rgb).latent_dist.sample()
            kl = None
        rec = vae.decode(z).sample                              # [B,1,64,768]

        l1 = (rec - bw).abs().mean()
        hl, _ce, _feat = htr_loss(rec, bw, text, tlen, tmask, pad)
        pyr = rec.new_zeros(())
        if pyr_kernel is not None:
            pyr = laplacian_pyramid_loss(rec, bw, pyr_kernel, args.pyramid_levels, pyr_alphas)
        nll = l1 + args.htr_weight * hl + args.pyramid_weight * pyr   # base L1 + htr + 피라미드(c)
        if log_var is not None:
            nll = nll / torch.exp(log_var) + log_var         # 학습형 uncertainty weighting(원본 AutoencoderLoss)
        loss = nll
        if kl is not None:
            loss = loss + args.kl_weight * kl
        (loss / args.grad_accum).backward()

        run["loss"] += loss.item(); run["l1"] += l1.item(); run["htr"] += hl.detach().item()
        run["pyr"] += float(pyr.detach()); run["kl"] += kl.detach().item() if kl is not None else 0.0
        run["n"] += 1
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_((p for p in vae.parameters() if p.requires_grad), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                n = max(1, run["n"]); sps = (time.time() - t0) / args.log_every
                lv = f"  lv={float(log_var.detach()):.3f}" if log_var is not None else ""
                pys = f"  pyr={run['pyr']/n:.4f}" if pyr_kernel is not None else ""
                print(f"step {step}/{args.max_steps}  loss={run['loss']/n:.4f}  l1={run['l1']/n:.4f}  "
                      f"htr={run['htr']/n:.4f}{pys}  kl={run['kl']/n:.1f}{lv}  {sps:.2f}s/step")
                tw.writerow([step, run["loss"]/n, run["l1"]/n, run["htr"]/n, run["pyr"]/n, run["kl"]/n, sps]); tl.flush()
                run = {"loss": 0.0, "l1": 0.0, "htr": 0.0, "pyr": 0.0, "kl": 0.0, "n": 0}; t0 = time.time()
            if args.val_every > 0 and step % args.val_every == 0:
                mse, sv = evaluate_vae(vae, val_bw, device)
                mse_en, sv_en = evaluate_vae(vae, val_en, device)
                print(f"  [val@{step}] KO MSE={mse:.5f} SSIM={sv:.4f} | EN MSE={mse_en:.5f} SSIM={sv_en:.4f}")
                vw.writerow([step, mse, sv, mse_en, sv_en]); vl.flush()
                dump_recon(vae, val_bw, device, out / "samples" / f"recon_s{step}.png")
            if step % args.save_every == 0:
                save(step)
    save(step)
    tl.close(); vl.close()
    print("done.")


if __name__ == "__main__":
    main()
