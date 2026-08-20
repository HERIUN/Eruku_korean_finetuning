"""한글 HTR finetune (VAE 한글 적응의 가독성 신호원).

원본 emuru VAE 학습의 가독성 핵심 = frozen HTR(OCR) loss. 그 HTR 은 라틴 전용(alphabet 167).
여기서 pretrained HTR 을 **warm-start** 로 한글에 finetune 한다:
  - `text_embedding`(입력)·`fc`(출력)를 한글 alphabet 크기로 **확장**하되, 라틴 인덱스(0..166)는
    pretrained 가중치 그대로 유지(custom_datasets.korean.alphabet 인덱스 규약) → 라틴 지식 전이.
  - feature_extractor/transformer 는 pretrained 그대로 이어서 학습.
과제: bw(clean 1ch) 이미지 → 텍스트(AR 디코딩). loss=SmoothCE + NoisyTeacherForcing. metric=CER.

결과는 `save_pretrained(out/htr_sXXXX)` → train/vae_korean.py 가 `HTR.from_pretrained` 로 로드.

예:
  python train/aux_htr.py --out finetune_runs/aux_htr_ko --max-steps 20000 --batch-size 64 --lr 1e-4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))

from models.htr import HTR
from models.smooth_ce import SmoothCrossEntropyLoss
from models.teacher_forcing import NoisyTeacherForcing
from custom_datasets.korean.alphabet import get_korean_alphabet
from custom_datasets.korean.aux import KoreanAuxDataset, aux_collate


def load_pretrained_htr(path: Path) -> HTR:
    """release(config.json + model.safetensors) 또는 save_pretrained 출력 둘 다 로드."""
    cfg = json.load(open(path / "config.json"))
    cfg = {k: v for k, v in cfg.items()
           if not k.startswith("_") and k not in ("architectures", "model_type", "torch_dtype", "transformers_version")}
    htr = HTR(**cfg)
    wf = path / "model.safetensors"
    if not wf.exists():
        wf = path / "diffusion_pytorch_model.safetensors"
    htr.load_state_dict(load_file(str(wf)), strict=False)
    return htr


def expand_htr_vocab(htr: HTR, new_size: int):
    """text_embedding + fc 를 new_size 로 확장. 기존 행(라틴/extra)은 보존, 신규(한글)만 랜덤 init."""
    old = htr.text_embedding.num_embeddings
    if old == new_size:
        return
    d = htr.d_model
    ne = nn.Embedding(new_size, d)
    nn.init.normal_(ne.weight, std=0.02)
    ne.weight.data[:old] = htr.text_embedding.weight.data
    htr.text_embedding = ne
    nf = nn.Linear(d, new_size)
    nn.init.normal_(nf.weight, std=0.02); nn.init.zeros_(nf.bias)
    nf.weight.data[:old] = htr.fc.weight.data
    nf.bias.data[:old] = htr.fc.bias.data
    htr.fc = nf
    htr.register_to_config(alphabet_size=new_size)  # save_pretrained 에 반영
    print(f"  vocab expanded {old} -> {new_size} (라틴/extra {old} 행 보존)")


def _lev(a, b):
    """Levenshtein 거리(문자 단위)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    tot_e, tot_n = 0, 0
    for p, r in zip(preds, refs):
        tot_e += _lev(p, r); tot_n += max(1, len(r))
    return tot_e / max(1, tot_n)


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="finetune_runs/aux_htr_ko")
    p.add_argument("--htr-pretrained", default="model_zoo/emuru_vae_htr")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise-prob", type=float, default=0.1, help="NoisyTeacherForcing 노이즈 확률(원본 0.1)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None, help="htr_sXXXX 디렉토리(가중치+trainer.pt)")
    return p


def main():
    args = make_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    alpha = get_korean_alphabet()
    A = len(alpha)
    print(f"korean alphabet_size = {A}")

    # 모델: resume 면 그 디렉토리, 아니면 pretrained warm-start
    src = Path(args.resume) if args.resume else Path(args.htr_pretrained)
    htr = load_pretrained_htr(src)
    expand_htr_vocab(htr, A)                    # resume 면 이미 A → no-op
    htr = htr.to(device)
    n = sum(p.numel() for p in htr.parameters()) / 1e6
    print(f"HTR params {n:.1f}M  (in_channels={htr.config.in_channels})")

    optimizer = torch.optim.Adam(htr.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
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
            start_step = int(st.get("step", 0))
            print(f"  resume step={start_step}")

    total = args.max_steps * args.grad_accum * args.batch_size
    ds = KoreanAuxDataset(length=total, seed=args.seed + start_step)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=aux_collate, drop_last=True, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    smooth_ce = SmoothCrossEntropyLoss(tgt_pad_idx=alpha.pad)
    noisy = NoisyTeacherForcing(A, alpha.num_extra_tokens, args.noise_prob)

    tl = (out / "htr_train.csv").open("a", newline="", encoding="utf-8")
    tw = csv.writer(tl)
    if tl.tell() == 0:
        tw.writerow(["step", "loss", "cer", "sec_per_step"])

    def save(step):
        d = out / f"htr_s{step}"
        htr.save_pretrained(d)
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, d / "trainer.pt")
        print(f"  saved {d}")

    print(f"training {start_step} -> {args.max_steps}")
    htr.train()
    step, micro = start_step, 0
    optimizer.zero_grad(set_to_none=True)
    run = {"loss": 0.0, "cer": 0.0, "n": 0}
    t0 = time.time()
    it = iter(loader)
    while step < args.max_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        img = b["bw"].to(device)
        text = b["text_logits_s2s"].to(device)
        tlen = b["texts_len"].to(device)
        tmask = b["tgt_key_mask"].to(device)
        pad = b["tgt_key_padding_mask"].to(device)

        noisy_text = noisy(text, tlen)
        out_logits = htr(img, noisy_text[:, :-1], tmask, pad[:, :-1])
        loss = smooth_ce(out_logits, text[:, 1:])
        (loss / args.grad_accum).backward()

        with torch.no_grad():
            pred = out_logits.argmax(-1)
            pc = alpha.decode(pred, [alpha.eos])
            rc = alpha.decode(text[:, 1:], [alpha.eos])
            c = cer(pc, rc)
        run["loss"] += loss.item(); run["cer"] += c; run["n"] += 1
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(htr.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                nn_ = max(1, run["n"]); sps = (time.time() - t0) / args.log_every
                print(f"step {step}/{args.max_steps}  loss={run['loss']/nn_:.4f}  cer={run['cer']/nn_:.4f}  {sps:.2f}s/step")
                tw.writerow([step, run["loss"]/nn_, run["cer"]/nn_, sps]); tl.flush()
                run = {"loss": 0.0, "cer": 0.0, "n": 0}; t0 = time.time()
            if step % args.save_every == 0:
                save(step)
    save(step)
    tl.close()
    print("done.")


if __name__ == "__main__":
    main()
