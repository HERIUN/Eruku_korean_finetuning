"""docs/MODULE_IO.md 용 그림 생성 — "학습 모듈에 뭐가 들어가고 뭐가 나오는가" 를 실제 이미지로.

학습 루프(train.core.train)와 **완전히 같은 호출**(split_collate → get_model_inputs → forward)을
한 배치에 대해 돌리고, 각 단계의 텐서를 이미지로 저장한다.

재현:
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python docs/_gen_module_io.py \
      --ckpt finetune_runs/korean_en2ko_fixed/checkpoint_step_014000.pth

출력: docs/img_module_io/*.png + numbers.json (md 본문 수치의 출처)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from models.eruku import pad_images                      # noqa: E402
from infer.show import crop_gen, label_img, load_model           # noqa: E402
from custom_datasets.korean_split import make_dataset, split_collate, to_gray  # noqa: E402

OUT = HERE / "img_module_io"


def _bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _label_row(text, w, h=24, bold=False, bg=240):
    """한글 라벨 줄 → BGR."""
    return _bgr(label_img(text, w, h, 15, bold=bold, center=False, bg=bg))


def _stack(blocks, pad=6, bg=255):
    """폭이 다른 BGR 블록들을 왼쪽정렬로 세로 결합."""
    w = max(b.shape[1] for b in blocks)
    rows = []
    for b in blocks:
        if b.shape[1] < w:
            b = np.pad(b, ((0, 0), (0, w - b.shape[1]), (0, 0)), constant_values=bg)
        rows.append(b)
        rows.append(np.full((pad, w, 3), bg, np.uint8))
    return np.vstack(rows[:-1])


def _decode_latent(model, emb, device):
    """decoder_inputs_embeds 의 한 구간([L,8]) → VAE 디코딩 grayscale [64, L*8]."""
    lat = rearrange(emb, "w (c h) -> 1 c h w", c=1, h=8).to(device)
    img = model.vae.decode(lat.float()).sample
    im = ((img.clamp(-1, 1) + 1) / 2)[0]
    if im.shape[0] > 1:
        im = im.mean(0, keepdim=True)
    return (im[0].cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "finetune_runs/korean_en2ko_fixed/checkpoint_step_014000.pth"))
    ap.add_argument("--max-img-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    nums = {"ckpt": args.ckpt, "max_img_len": args.max_img_len}

    # ── 1. 데이터: 학습 때와 동일한 dataset + collate ───────────────────────────
    # 증강(회전/블러/jitter…)은 random·np.random·torch 세 RNG 를 모두 쓴다
    # (KoreanSplitFontSquare._snap/_restore). 문서 수치가 run 마다 달라지지 않게 셋 다 고정.
    ds = make_dataset(style_range=(2, 4), gen_range=(3, 6), length=10_000, seed=args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    samples = [ds[i] for i in (3, 11, 25)]
    batch = split_collate(samples)
    nums["batch"] = [
        {"style_text": batch["style_text"][i], "gen_text": batch["gen_text"][i],
         "style_img_shape": list(batch["style_img"][i].shape),
         "gen_img_shape": list(batch["gen_img"][i].shape),
         "style_img_len": int(batch["style_img_len"][i]),
         "gen_img_len": int(batch["gen_img_len"][i])}
        for i in range(len(samples))
    ]

    # 01 — 샘플 1개의 입력 원본 (style / gen)
    s0 = samples[0]
    si, gi = to_gray(s0["style_img"]), to_gray(s0["gen_img"])
    blocks = [
        _label_row(f"style_img  [3,64,{si.shape[1]}]  style_text = '{s0['style_text']}'",
                   max(si.shape[1], 700), bold=True),
        _bgr(si),
        _label_row(f"gen_img (=GT)  [3,64,{gi.shape[1]}]  gen_text = '{s0['gen_text']}'",
                   max(gi.shape[1], 700), bold=True),
        _bgr(gi),
    ]
    cv2.imwrite(str(OUT / "01_inputs.png"), _stack(blocks))

    # 02 — 왜 style_img_len / gen_img_len 이 따로 필요한가:
    #      배치는 pad_images(padding_value=1=흰색) 로 최대폭에 맞춰지고, 패딩은 여백과
    #      픽셀값이 같다 → 텐서만 봐선 진짜 끝을 못 찾는다. 빨간선 = 진짜 폭(*_img_len).
    padded = pad_images([el for el in batch["style_img"]])       # [B,3,64,Wmax]
    nums["padded_style_shape"] = list(padded.shape)
    rows = [_label_row("pad_images() 후 batch['style_img'] — 빨간선 = style_img_len (진짜 끝)",
                       padded.shape[-1], bold=True)]
    for i in range(padded.shape[0]):
        arr = _bgr(to_gray(padded[i]))
        cut = int(batch["style_img_len"][i])
        cv2.line(arr, (cut - 1, 0), (cut - 1, arr.shape[0] - 1), (0, 0, 255), 2)
        rows.append(_label_row(f"[{i}] style_img_len={cut}px  (텐서 폭은 {padded.shape[-1]}px)",
                               padded.shape[-1], bg=250))
        rows.append(arr)
    cv2.imwrite(str(OUT / "02_why_len.png"), _stack(rows))

    # ── 2. 모델 ────────────────────────────────────────────────────────────────
    model, step = load_model(args.ckpt, device)
    nums["step"] = step

    mi = model.get_model_inputs(batch["style_img"], batch["gen_img"],
                                batch["style_img_len"], batch["gen_img_len"], args.max_img_len)
    dec = mi["decoder_inputs_embeds"].to(device)      # [B, L, 8]
    sp = mi["specials"].to(device)                    # [B, L]
    nums["decoder_inputs_embeds_shape"] = list(dec.shape)
    nums["specials_shape"] = list(sp.shape)
    nums["lengths"] = mi["lengths"].tolist()

    # 03 — 디코더 시퀀스의 실제 내용: latent 를 되돌려 그리고, 아래에 specials 색띠
    emb0, sp0 = dec[0], sp[0]
    L0 = int(mi["lengths"][0])
    strip = _decode_latent(model, emb0[:L0], device)              # [64, L0*8]
    bar = np.zeros((18, L0 * 8, 3), np.uint8)
    color = {2: (60, 60, 60), 0: (0, 160, 0), 1: (255, 0, 0), 1000: (200, 200, 200)}
    for j in range(L0):
        bar[:, j * 8:(j + 1) * 8] = color.get(int(sp0[j].item()), (200, 200, 200))
    sog = (sp0[:L0] == 0).nonzero().flatten()
    nums["sample0"] = {
        "seq_len_latent": L0, "seq_len_px": L0 * 8,
        "sog_at": int(sog[0].item()) if len(sog) else None,
        "n_img_tokens": int((sp0[:L0] == 2).sum().item()),
    }
    cv2.imwrite(str(OUT / "03_decoder_seq.png"), _stack([
        _label_row("decoder_inputs_embeds[0] 를 VAE 로 되돌린 그림 = 모델이 실제로 받는 시퀀스",
                   strip.shape[1], bold=True),
        _bgr(strip),
        _label_row("specials[0]:  회색=Img(2)  초록=SOG(0)  파랑=EOG(1)", strip.shape[1], bg=250),
        bar,
    ]))

    # ── 3. 출력 (a) teacher forcing 한 스텝 ────────────────────────────────────
    with torch.no_grad():
        losses, pred_latent = model.forward(decoder_inputs_embeds_vae=dec, specials=sp,
                                            style_text=batch["style_text"], gen_text=batch["gen_text"])
    nums["losses"] = {k: float(v) for k, v in losses.items()}

    # loss 가 '무엇 위에서' 계산되는지 수치로 남긴다 (md 5.1 절 근거)
    B, L = sp.shape
    mask = (sp == 2).unsqueeze(2)                                  # Img 열만
    pred_seq = pred_latent.permute(0, 3, 2, 1).squeeze(-1)         # [B,1,8,L] → [B,L,8]
    se = (pred_seq * mask - dec * mask) ** 2
    n_all, n_img = se.numel(), int(mask.expand_as(se).sum().item())
    sog0 = int((sp[0] == 0).nonzero().flatten()[0].item())
    nums["mse_detail"] = {
        "cols_total": B * L,
        "cols_img(specials==2)": int((sp == 2).sum().item()),
        "cols_sog(0)": int((sp == 0).sum().item()),
        "cols_eog_or_pad(1)": int((sp == 1).sum().item()),
        "sample0_style_cols_labeled_img": int((sp[0, :sog0] == 2).sum().item()),
        "sample0_style_cols_total": sog0,
        "pad_label_is_eog": all(set(sp[i, int(mi["lengths"][i]):].tolist()) <= {1}
                                for i in range(B)),
        "mse_as_coded(전체 평균)": float(losses["mse_loss"]),
        "mse_if_divided_by_img_only": float(se.sum() / max(1, n_img)),
        "img_element_ratio": n_img / n_all,
    }

    # pred_latent[..., i] 는 입력 x_i 의 예측이다(디코더 입력에 sos 를 붙이고 logits[:, :-1] 를
    # 쓰므로 이미 정렬돼 있다) → GT 와 같은 인덱스로 잘라야 짝이 맞는다.
    def _dec_lat(lat):
        im = ((model.vae.decode(lat.float()).sample.clamp(-1, 1) + 1) / 2)[0]
        if im.shape[0] > 1:
            im = im.mean(0, keepdim=True)
        return (im[0].cpu().numpy() * 255).astype(np.uint8)

    pred_seq0 = pred_seq[0]                                        # [L,8] (GT dec[0] 과 같은 인덱스)
    g0, g1 = sog0 + 1, L0 - 1                                      # gen 구간(SOG 다음 ~ EOG 앞)
    to_lat = lambda e: rearrange(e, "w (c h) -> 1 c h w", c=1, h=8).to(device)
    gt_gen = _dec_lat(to_lat(dec[0, g0:g1]))
    pr_gen = _dec_lat(to_lat(pred_seq0[g0:g1]))
    pr_all = _dec_lat(to_lat(pred_seq0[:L0]))
    nums["sample0"]["gen_cols"] = [g0, g1]

    # 선명도 차이의 출처: 학습 GT 는 gen_img 원본픽셀이 아니라 'VAE 를 한 번 왕복한 것'.
    # 그런데 그 왕복은 **배치 최대폭으로 패딩된 이미지**를 encode 한 결과다 → 같은 이미지라도
    # 배치에 긴 샘플이 있으면 latent 가 나빠진다(VAE 의 GroupNorm 이 패딩 흰 영역까지 포함해
    # 전역 통계를 잡기 때문). 아래에서 그 기여를 분리해 수치로 남긴다.
    gw = min(gi.shape[1], gt_gen.shape[1])
    orig_g = gi[:, :gw].astype(np.float32) / 255.0
    rt_g = gt_gen[:, :gw].astype(np.float32) / 255.0
    gen_px = int(batch["gen_img_len"][0])

    @torch.no_grad()
    def _roundtrip_padded(W, how="mode"):
        """gen_img 를 폭 W 로 흰색 패딩해 encode → gen 구간만 잘라 디코딩."""
        x = torch.ones(1, 3, 64, max(W, gen_px))
        x[0, :, :, :gen_px] = s0["gen_img"]
        d = model.vae.encode(x.to(device).float()).latent_dist
        z = d.mode() if how == "mode" else d.sample()
        return _dec_lat(z[:, :, :, :gen_px // 8])

    def _m(a, b):
        w = min(a.shape[1], b.shape[1])
        return float(((a[:, :w].astype(np.float32) / 255.0 - b[:, :w].astype(np.float32) / 255.0) ** 2).mean())

    # 주의: gen 은 gen 끼리 패딩된다 → 기준 폭은 style 배치폭이 아니라 gen 배치 최대폭
    pad_w = max(int(el) for el in batch["gen_img_len"])
    sweep = {str(W): _m(gi, _roundtrip_padded(W)) for W in (gen_px, 500, 700, pad_w, 1400)}
    nums["vae_roundtrip"] = {
        "compare_px": int(gw),
        "mse_orig_vs_gt(학습경로=패딩 encode)": float(((orig_g - rt_g) ** 2).mean()),
        "mse_orig_vs_unpadded(패딩 없이 단독 encode)": _m(gi, _roundtrip_padded(gen_px)),
        "mse_gt_vs_pred": float(((rt_g - pr_gen[:, :gw].astype(np.float32) / 255.0) ** 2).mean()),
        "pad_width_sweep(mode)": sweep,
        "sample_vs_mode_at_unpadded": {
            "mode": _m(gi, _roundtrip_padded(gen_px, "mode")),
            "sample": _m(gi, _roundtrip_padded(gen_px, "sample")),
        },
        "note": "패딩 폭이 넓을수록 같은 이미지의 재구성이 나빠진다. sample()/mode() 차이는 무시할 수준.",
    }

    # (a) 입력 gen_text 와 짝이 맞는 부분만 — 원본 → VAE 왕복(=GT) → 예측
    W4 = max(gi.shape[1], gt_gen.shape[1], 700)
    cv2.imwrite(str(OUT / "04_teacher_forced.png"), _stack([
        _label_row(f"gen_text = '{s0['gen_text']}'  (gen 구간 = SOG 다음 ~ EOG 앞, latent 열 {g0}~{g1})",
                   W4, bold=True),
        _label_row(f"1) 원본 gen_img (데이터 픽셀, {gi.shape[1]}px)", W4, bg=250),
        _bgr(gi),
        _label_row(f"2) GT = 1)을 배치패딩({pad_w}px) 상태로 VAE 왕복한 것 = 모델이 맞춰야 할 실제 정답 "
                   f"— 1)과의 MSE {nums['vae_roundtrip']['mse_orig_vs_gt(학습경로=패딩 encode)']:.4f} "
                   f"(패딩 없이 왕복하면 {nums['vae_roundtrip']['mse_orig_vs_unpadded(패딩 없이 단독 encode)']:.4f})",
                   W4, bg=250),
        _bgr(gt_gen),
        _label_row(f"3) PRED (pred_latent[0] 의 같은 구간) — 2)와의 MSE "
                   f"{nums['vae_roundtrip']['mse_gt_vs_pred']:.4f}", W4, bg=250),
        _bgr(pr_gen),
    ]))

    # (b) 전체 시퀀스 — style 구간도 loss 에 들어간다는 것을 보이기 위해
    cv2.imwrite(str(OUT / "04b_teacher_forced_full.png"), _stack([
        _label_row("GT 전체 시퀀스 = [style | SOG | gen | EOG]", strip.shape[1], bold=True),
        _bgr(strip),
        _label_row("PRED 전체 — style 구간도 함께 예측·학습된다 (mse mask = specials==2 전부)",
                   pr_all.shape[1], bold=True),
        _bgr(pr_all),
    ]))

    # ── 4. 출력 (b) 실제 자기회귀 생성 (gen_img 없이 style 만으로) ──────────────
    st_img = s0["style_img"]
    mi_s = model.get_model_inputs([st_img], None, style_len=int(st_img.shape[-1]),
                                  gen_len=None, max_img_len=args.max_img_len)
    dec_s = mi_s["decoder_inputs_embeds"].to(device)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        img, special = model.generate(dec_s, [s0["style_text"]], [s0["gen_text"]],
                                      cfg_scale=args.cfg, max_new_tokens=args.max_new_tokens)
    gen_arr = crop_gen(img[0], special, dec_s.shape[1] * 8)
    nums["generate"] = {"cfg": args.cfg, "prefix_latent": int(dec_s.shape[1]),
                        "out_px": int(gen_arr.shape[1])}
    cv2.imwrite(str(OUT / "05_generated.png"), _stack([
        _label_row(f"입력 style_img (조건) + style_text='{s0['style_text']}'",
                   max(si.shape[1], 700), bold=True),
        _bgr(si),
        _label_row(f"목표 gen_text='{s0['gen_text']}' → 자기회귀 생성 (cfg={args.cfg}, gen_img 없음)",
                   max(gen_arr.shape[1], 700), bold=True),
        _bgr(gen_arr),
        _label_row("참고: 같은 폰트의 GT (gen_img) — 학습 때만 존재", max(gi.shape[1], 700), bg=250),
        _bgr(gi),
    ]))

    (OUT / "numbers.json").write_text(json.dumps(nums, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(nums, ensure_ascii=False, indent=2))
    print(f"\nsaved → {OUT}")


if __name__ == "__main__":
    main()
