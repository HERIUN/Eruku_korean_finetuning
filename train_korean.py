"""Eruku 한글화 minimal training launcher.

- Emuru 모델 (T5-base + VAE + OrigamiNet)
- ByT5 tokenizer (한글 byte 자동 처리)
- VAE: `blowing-up-groundhogs/emuru_vae` (HF, download)
- T5: from-scratch (영어 pretrained 안 받음)
- OrigamiNet: alpha=1.0 → ocr_loss 사용 안 함 → 한글에서도 동작

Usage:
  python train_korean.py --lines-json ../font_ai_pipeline_work/benchmark/train_lines.json --out finetune_runs/korean --max-steps 20000
"""
from __future__ import annotations

import argparse
import datetime
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_handb_korean",
    HERE / "custom_datasets" / "real_datasets" / "handb_korean.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HanDBKoreanDataset = _mod.HanDBKoreanDataset
handb_collate = _mod.handb_collate

from eruku_continuous_inf import Emuru


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lines-json", default=None, help="offline lines_json (online-split 시 불필요)")
    p.add_argument("--out", default="finetune_runs/korean")
    p.add_argument("--vae-checkpoint", default="blowing-up-groundhogs/emuru_vae")
    p.add_argument("--t5-checkpoint", default="google-t5/t5-large")
    p.add_argument("--eruku-pretrained", default="model_zoo/eruku_pretrained/000073688.pth",
                   help="영어 pretrained ckpt. 로컬에 없으면 HF blowing-up-groundhogs/eruku "
                        "의 pytorch_model.bin 을 자동 다운로드 (~2.9GB)")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-img-len", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ocr-checkpoint", default=None,
                   help="OrigamiNet ckpt (기본 None=미사용. alpha<1 로 OCR loss 쓸 때만 필요)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-dropout", type=float, default=0.1,
                   help="학습 중 text 를 uncond 로 drop 할 확률 (>0 이어야 inference CFG 동작)")
    p.add_argument("--resume", default=None,
                   help="체크포인트에서 이어 학습 (model+optimizer+step 복원, eruku-pretrained 무시)")
    p.add_argument("--style-text-dropout", type=float, default=0.0,
                   help="논문 p_drop (Phase2): style text(T_s)만 비울 확률. gen text 는 유지. 논문 0.10")
    p.add_argument("--online-split", action="store_true",
                   help="repo OnlineSplitFontSquare 온라인 한글 split 데이터 사용 (lines-json 대신)")
    p.add_argument("--style-words", nargs=2, type=int, default=[1, 8], help="online: style 어절수 범위")
    p.add_argument("--gen-words", nargs=2, type=int, default=[1, 32], help="online: gen(target) 어절수 범위")
    p.add_argument("--save-samples", type=int, default=0, help="학습 시작 시 데이터 샘플 N개 저장")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── run config 기록: train_config.yml 에 run 히스토리 누적 ──
    cfg_path = out_dir / "train_config.yml"
    runs = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or []) if cfg_path.exists() else []
    cur = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}
    if args.resume and runs:                      # resume 인데 파라미터가 달라지면 경고
        prev = runs[-1]["args"]
        for k in cur:
            if k not in ("resume", "max_steps") and prev.get(k) != cur[k]:
                print(f"[config WARN] resume 인데 '{k}' 가 이전 run 과 다름: {prev.get(k)!r} -> {cur[k]!r}")
    runs.append({"run": len(runs) + 1,
                 "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
                 "torch": str(torch.__version__),
                 "args": cur})
    cfg_path.write_text(yaml.safe_dump(runs, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"config logged: {cfg_path} (run #{len(runs)})")

    # ── dataset ──
    if args.online_split:
        from korean_split_dataset import make_dataset, split_collate, dump_samples
        dataset = make_dataset(style_range=tuple(args.style_words),
                               gen_range=tuple(args.gen_words),
                               length=args.max_steps * args.batch_size, seed=args.seed)
        if args.save_samples > 0:
            dump_samples(dataset, args.save_samples, out_dir / "train_samples")
        collate = split_collate
    else:
        dataset = HanDBKoreanDataset(args.lines_json, img_height=64, batch_keys=["style", "same"])
        collate = handb_collate
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
    )

    # ── model ──
    # OCR(OrigamiNet) 은 alpha=1.0 이라 미사용 → ocr_checkpoint=None 으로 생략 (기본)
    model = Emuru(
        t5_checkpoint=args.t5_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        ocr_checkpoint=args.ocr_checkpoint,
    ).to(device)
    model.alpha = 1.0  # ocr_loss off
    # text-dropout 켜서 CFG 가능하게 (원본 Emuru 레시피). 이게 없으면 generate() 의
    # CFG 가 no-op (uncond==cond) 이라 inference 에서 텍스트를 강하게 못 따라감.
    model.dropout_probability = args.text_dropout
    model.drop_text = args.text_dropout > 0
    print(f"text-dropout: prob={model.dropout_probability} drop_text={model.drop_text}")
    model.set_training(model.T5, True)
    model.set_training(model.vae, False)
    if model.ocr is not None:
        model.set_training(model.ocr, False)

    # 영어 학습 Eruku ckpt 로드 (strict=False). resume 시엔 건너뜀(resume ckpt 가 모델).
    # 로컬 파일이 없으면 HF 공식 릴리즈(pytorch_model.bin, ~2.9GB)를 자동 다운로드.
    if not args.resume and args.eruku_pretrained:
        ck_path = Path(args.eruku_pretrained)
        if not ck_path.exists():
            from huggingface_hub import hf_hub_download
            print("local pretrained 없음 → HF 자동 다운로드: blowing-up-groundhogs/eruku (~2.9GB)")
            ck_path = Path(hf_hub_download("blowing-up-groundhogs/eruku", "pytorch_model.bin"))
        print(f"loading pretrained: {ck_path}")
        ckpt = torch.load(ck_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"  missing sample: {missing[:3]}")
        del ckpt, state

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=1e-2,
    )

    start_step = 0
    if args.resume and Path(args.resume).exists():
        print(f"resuming from: {args.resume}")
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        rstate = rck["model"] if "model" in rck else rck
        if any(k.startswith("module.") for k in rstate):
            rstate = {k[len("module."):]: v for k, v in rstate.items()}
        m, u = model.load_state_dict(rstate, strict=False)
        print(f"  model missing={len(m)} unexpected={len(u)}")
        if "optimizer" in rck:
            try:
                optimizer.load_state_dict(rck["optimizer"])
                print("  optimizer state restored")
            except Exception as e:
                print(f"  optimizer restore 실패(무시): {e}")
        start_step = int(rck.get("step", 0))
        print(f"  resume step={start_step} (→ max-steps {args.max_steps})")
        del rck, rstate

    # ── loss 로그: train_loss.csv (append, resume 시 이어 씀) ──
    loss_csv = out_dir / "train_loss.csv"
    loss_f = open(loss_csv, "a", buffering=1)
    if loss_f.tell() == 0:
        loss_f.write("step,loss,mse,ce,it_s,timestamp\n")
    print(f"loss logged: {loss_csv}")

    model.train()
    step = start_step
    skipped = 0
    t0 = time.time()
    last_log = t0
    losses_window = []

    while step < args.max_steps:
        for sample in loader:
            if step >= args.max_steps:
                break

            # sample = list 들 (handb_collate)
            try:
                model_inputs = model.get_model_inputs(
                    sample["style_img"], sample["same_img"],
                    sample["style_img_len"], sample["same_img_len"],
                    args.max_img_len,
                )
            except Exception as e:
                print(f"[step {step}] get_model_inputs ERR: {e}")
                continue

            decoder_inputs_embeds_vae = model_inputs["decoder_inputs_embeds"].to(device)
            specials = model_inputs["specials"].to(device)

            # forward 의 specials/embeds seq 길이가 batch 에 따라 1 어긋나는
            # 간헐 shape 버그가 있어 (eruku forward 내부) 드문 malformed batch 는 skip.
            try:
                # style-text dropout (논문 p_drop, Phase2): style text(T_s)만 비움, gen text(T_g) 유지
                style_text = sample["style_text"]
                if args.style_text_dropout > 0:
                    style_text = ["" if random.random() < args.style_text_dropout else s
                                  for s in style_text]
                out = model.forward(
                    decoder_inputs_embeds_vae=decoder_inputs_embeds_vae,
                    specials=specials,
                    style_text=style_text,
                    gen_text=sample["same_text"],
                )
                losses, pred_latent = out
                loss = losses["loss"]
                mse = losses["mse_loss"]
                ce = losses["ce_loss"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except Exception as e:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                print(f"[step {step}] forward/backward SKIP ({skipped} total): {e}")
                continue

            losses_window.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses_window[-args.log_every:]) / min(len(losses_window), args.log_every)
                ips = args.log_every / (time.time() - last_log + 1e-9)
                last_log = time.time()
                print(f"step {step}/{args.max_steps}  loss={avg:.4f}  mse={mse.item():.4f}"
                      f"  ce={ce.item():.4f}  {ips:.1f} it/s")
                loss_f.write(f"{step},{avg:.6f},{mse.item():.6f},{ce.item():.6f},"
                             f"{ips:.2f},{datetime.datetime.now().isoformat(timespec='seconds')}\n")

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
                p = out_dir / f"checkpoint_step_{step:06d}.pth"
                torch.save(ckpt, p)
                print(f"  saved {p}")

    final = out_dir / "checkpoint_last.pth"
    torch.save({"model": model.state_dict(), "step": step}, final)
    loss_f.close()
    print(f"done. saved {final}  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
