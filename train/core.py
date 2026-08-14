"""Eruku 한글화 학습 공통 코어.

Phase 1 / Phase 2 가 공유하는 인자 정의(`make_parser`)와 학습 루프(`train`).
- Phase 1 진입점: train/phase1.py
- Phase 2 진입점: train/phase2.py
- 영어→한글 직접: train/korean_from_en.py

모델: Emuru(T5-large + ByT5 tokenizer + 동결 VAE + 동결 OrigamiNet, alpha=1.0 → OCR loss off)
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

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
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


def make_parser():
    """Phase 1/2 가 공유하는 전체 인자. 각 Phase 진입점이 set_defaults 로 기본값을 덮어쓴다."""
    p = argparse.ArgumentParser()
    p.add_argument("--lines-json", default=None, help="offline lines_json (online-split 시 불필요)")
    p.add_argument("--out", default="finetune_runs/korean")
    p.add_argument("--vae-checkpoint", default="blowing-up-groundhogs/emuru_vae")
    p.add_argument("--jamo-text", action="store_true",
                   help="C1: 텍스트 조건을 NFD 자모로 분해해서 넣는다(겹받침 개선 실험). "
                        "추론에서도 반드시 같은 설정이어야 하므로 ckpt 에 함께 저장된다. "
                        "※ A/B 기각됨(docs/JONGSEONG_STROKE_LOSS.md 실험 1) — 재현용")
    p.add_argument("--reset-vae", action="store_true",
                   help="resume/pretrained 로드 시 ckpt 의 vae.* 키를 strip → --vae-checkpoint 로 준 새 VAE 유지. "
                        "한글적응 full VAE 교체 후 Eruku 재적응 시 필수(안 쓰면 ckpt 의 옛 VAE 가 덮어씀)")
    p.add_argument("--t5-checkpoint", default="google-t5/t5-large")
    p.add_argument("--eruku-pretrained", default="model_zoo/eruku_pretrained/000073688.pth",
                   help="영어 pretrained ckpt. 로컬에 없으면 HF blowing-up-groundhogs/eruku "
                        "의 pytorch_model.bin 을 자동 다운로드 (~2.9GB)")
    p.add_argument("--max-steps", type=int, default=20000,
                   help="누적 virtual step 상한 (resume 시에도 절대값). --extra-steps 지정 시 무시됨")
    p.add_argument("--extra-steps", type=int, default=None,
                   help="resume step 에 더해서 학습할 step 수. 지정 시 max-steps = resume_step + extra-steps "
                        "로 자동계산 (Phase 2 의 '5000 step 더' 를 max-steps 누적값 실수 없이 표현)")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation 횟수. virtual batch = batch-size × grad-accum. "
                        "논문: virtual 256 (예: batch 2 × accum 128) + lr 1e-4. "
                        "max-steps/save-every/log-every 는 virtual step 기준")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-img-len", type=int, default=8192,
                   help="디코더 시퀀스 최대 픽셀폭(=//8 토큰). 원본 Eruku=8192. "
                        "2048 이면 긴 gen 의 71%%가 잘려 조기 EOG 발생(측정확인). 95GB VRAM 이면 8192 가능")
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
    p.add_argument("--english-fonts-dir", default=str(HERE / "assets" / "fonts_english"),
                   help="라틴(영어) 폰트 디렉토리. 영어 전용 데이터 writer 풀")
    p.add_argument("--english-frac", type=float, default=0.0,
                   help="학습 샘플 중 영어 전용(라틴 폰트) 비율(0=미사용). 한글폰트는 영어도 렌더하지만 "
                        "라틴 폰트로 영어 스타일 다양성을 추가 노출 → 영어 fidelity·획 다양성 보강")
    p.add_argument("--legacy-transform", action="store_true",
                   help="원본 composite 데이터 transform([style|gap|gen] 합쳐 렌더+전체증강+경계분할) 사용. "
                        "데이터 정책 변경(gen 흰배경·곧은글자) 전후 비교 실험용")
    p.add_argument("--save-samples", type=int, default=0, help="학습 시작 시 데이터 샘플 N개 저장")
    # ── validation (held-out 폰트에서 val loss 측정) ──
    p.add_argument("--val-every", type=int, default=0,
                   help="N virtual step 마다 val loss 측정(0=비활성). "
                        "online-split 일 때만 동작. val_loss.csv 에 기록")
    p.add_argument("--val-fonts-dir", default=None,
                   help="val 한글 폰트 디렉토리. 기본(None)=--val-font-frac 으로 학습 폰트에서 분할. "
                        "held-out 일반화를 보려면 assets/fonts_korean_v2/test 지정 "
                        "(test 폰트는 최종 평가 전용 권장)")
    p.add_argument("--val-font-frac", type=float, default=0.1,
                   help="--val-fonts-dir 미지정 시 학습 폰트에서 val 로 떼는 비율(val_seed 로 결정적 분할). "
                        "val 폰트는 학습에서 제외 → 폰트 일반화 측정. 0=분할 안 함(전체 폰트로 학습, "
                        "val 은 같은 폰트에서 텍스트만 val_seed 로 분리)")
    p.add_argument("--val-english-fonts-dir", default=None,
                   help="val 영어(라틴) 폰트 디렉토리. 미지정 시 --english-fonts-dir 사용. "
                        "(전용 held-out 영어 폰트 디렉토리를 만들면 여기에 지정)")
    p.add_argument("--val-samples", type=int, default=0,
                   help="val 세트 총 샘플 수. 0=하위호환(val-batches × batch-size). "
                        "이 안에서 한글:영어 비율은 --english-frac 그대로(학습과 동일).")
    p.add_argument("--val-batches", type=int, default=4,
                   help="(하위호환) --val-samples 미지정 시 val 샘플수 = val-batches × batch-size")
    p.add_argument("--val-seed", type=int, default=1234,
                   help="val 세트 생성 시드(고정 → step 간 동일 샘플로 공정 비교)")
    return p


def truncation_reason(sl_px, gl_px, max_img_len):
    """get_model_inputs 의 clamp 와 **동일한** 계산으로 이 샘플이 잘리는지 판정.

    잘림 조건(둘 중 하나라도):
      - style_len > max_img_len//2           → style 이 예산 절반에 잘림
      - gen_len   > max_img_len - style - 16 → 남은 예산(SOG/EOG 16px 제외)에 gen 이 잘림
    (단위: 모두 픽셀. get_model_inputs 는 latent*8 로 픽셀 환산해 비교 → 여기도 픽셀.)

    반환: None(안전) | (kind, have_px, budget_px)  kind ∈ {"style","gen"}
    """
    sl = min(int(sl_px), max_img_len // 2)
    if int(sl_px) > sl:
        return ("style", int(sl_px), max_img_len // 2)
    remaining = max_img_len - sl - 16
    if int(gl_px) > remaining:
        return ("gen", int(gl_px), remaining)
    return None


def filter_no_truncation(sample, max_img_len):
    """배치에서 잘릴 샘플을 제거한다(절대 truncation 금지 정책).

    반환: (kept_sample[dict], dropped[list of (idx, kind, have, budget, style_text, gen_text)])
    kept_sample 은 원본과 같은 키(list)들만 남긴 사본. 전부 잘리면 리스트가 빈다.
    """
    keys = ["style_img", "gen_img", "style_text", "gen_text", "style_img_len", "gen_img_len"]
    n = len(sample["style_img"])
    keep_idx, dropped = [], []
    for i in range(n):
        r = truncation_reason(sample["style_img_len"][i], sample["gen_img_len"][i], max_img_len)
        if r is None:
            keep_idx.append(i)
        else:
            kind, have, budget = r
            dropped.append((i, kind, have, budget,
                            sample["style_text"][i], sample["gen_text"][i]))
    kept = {k: [sample[k][i] for i in keep_idx] for k in keys}
    return kept, dropped


@torch.no_grad()
def _evaluate(model, val_batches, device, max_img_len):
    """고정 held-out 배치들에서 평균 val loss(ce+mse) 계산. 학습 loss 와 동일 계산식."""
    model.eval()
    tot = {"loss": 0.0, "mse": 0.0, "ce": 0.0}
    n = 0
    for sample in val_batches:
        try:
            mi = model.get_model_inputs(
                sample["style_img"], sample["gen_img"],
                sample["style_img_len"], sample["gen_img_len"], max_img_len)
            dec = mi["decoder_inputs_embeds"].to(device)
            sp = mi["specials"].to(device)
            losses, _ = model.forward(
                decoder_inputs_embeds_vae=dec, specials=sp,
                style_text=sample["style_text"], gen_text=sample["gen_text"])
        except Exception as e:
            print(f"  [val skip] {e}")
            continue
        tot["loss"] += losses["loss"].item()
        tot["mse"] += losses["mse_loss"].item()
        tot["ce"] += losses["ce_loss"].item()
        n += 1
    model.train()
    if n == 0:
        return None
    return tot["loss"] / n, tot["mse"] / n, tot["ce"] / n


def _load_or_build_val_samples(args, out_dir, val_fonts=None):
    """고정 held-out val 세트를 만들거나 디스크에서 로드.

    - 총 개수는 --val-samples(없으면 val-batches × batch-size).
    - 그 안의 한글:영어 비율은 --english-frac(학습과 동일). 한글 폰트는 --val-fonts-dir
      (기본 None=학습 폰트, val_seed 로 텍스트만 분리), 영어는 --val-english-fonts-dir(기본: 학습 영어 폰트).
    - out_dir/val_set.pt 에 저장 → 이후 run 은 설정이 같으면 **재렌더 없이 로드**
      (매 run 새로 샘플링하던 문제 제거 = step/run 간 완전히 동일한 val).
    반환: (samples[list of dict], n_val)
    """
    from korean_split_dataset import make_dataset as _mk, make_english_dataset as _mken

    n_val = args.val_samples if args.val_samples > 0 else args.val_batches * args.batch_size
    en_fonts = args.val_english_fonts_dir or args.english_fonts_dir
    sig = {
        "n_val": n_val, "english_frac": float(args.english_frac),
        "style_words": list(args.style_words), "gen_words": list(args.gen_words),
        "val_seed": args.val_seed, "val_fonts_dir": args.val_fonts_dir,
        "val_font_names": sorted(p.name for p in val_fonts) if val_fonts else None,
        "val_english_fonts_dir": en_fonts,
    }
    cache = out_dir / "val_set.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if blob.get("sig") == sig:
            print(f"val set 로드(고정, 재렌더 없음): {cache} ({len(blob['samples'])} 샘플)")
            return blob["samples"], n_val
        print(f"val set 설정 변경 감지 → 재생성 ({cache})")

    en_n = int(round(n_val * args.english_frac)) if args.english_frac > 0 else 0
    ko_n = n_val - en_n

    def _draw_safe(ds, target):
        """잘리지 않는(truncation_reason=None) 샘플만 target 개 모은다. 잘림 샘플은 건너뛰고 재추첨."""
        out, skipped, i, cap = [], 0, 0, target * 50 + 200
        while len(out) < target and i < cap:
            s = ds[i]; i += 1
            if truncation_reason(int(s["style_img"].shape[-1]),
                                 int(s["gen_img"].shape[-1]), args.max_img_len) is None:
                out.append(s)
            else:
                skipped += 1
        return out, skipped

    samples, ko_skip, en_skip = [], 0, 0
    if ko_n > 0:
        kds = _mk(style_range=tuple(args.style_words), gen_range=tuple(args.gen_words),
                  length=ko_n * 60, seed=args.val_seed,
                  fonts_dir=(args.val_fonts_dir or val_fonts))
        got, ko_skip = _draw_safe(kds, ko_n); samples += got
    if en_n > 0:
        eds = _mken(style_range=tuple(args.style_words), gen_range=tuple(args.gen_words),
                    length=en_n * 60, seed=args.val_seed + 777, fonts_dir=en_fonts)
        got, en_skip = _draw_safe(eds, en_n); samples += got
    # deterministic shuffle → 배치마다 한글/영어가 섞이게(비율 편중 방지)
    random.Random(args.val_seed).shuffle(samples)
    torch.save({"sig": sig, "samples": samples}, cache)
    print(f"val set 생성·저장: {cache} (한글 {ko_n} + 영어 {en_n} = {len(samples)}개, "
          f"truncation 으로 재추첨한 수: ko={ko_skip} en={en_skip})")
    return samples, n_val


def train(args):
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
            if k not in ("resume", "max_steps", "extra_steps") and prev.get(k) != cur[k]:
                print(f"[config WARN] resume 인데 '{k}' 가 이전 run 과 다름: {prev.get(k)!r} -> {cur[k]!r}")
    runs.append({"run": len(runs) + 1,
                 "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
                 "torch": str(torch.__version__),
                 "args": cur})
    cfg_path.write_text(yaml.safe_dump(runs, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"config logged: {cfg_path} (run #{len(runs)})")

    # ── model ──
    # OCR(OrigamiNet) 은 alpha=1.0 이라 미사용 → ocr_checkpoint=None 으로 생략 (기본)
    model = Emuru(
        t5_checkpoint=args.t5_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        ocr_checkpoint=args.ocr_checkpoint,
    ).to(device)
    model.alpha = 1.0  # ocr_loss off
    model.jamo_text = args.jamo_text
    print(f"jamo-text(C1): {model.jamo_text}")
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
        if args.reset_vae:
            n0 = len(state); state = {k: v for k, v in state.items() if not k.startswith("vae.")}
            print(f"  [reset-vae] pretrained vae.* {n0-len(state)}키 strip → --vae-checkpoint 유지")
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

    # ── resume: model+optimizer+step 복원 (dataset 생성 전에 start_step 확정) ──
    start_step = 0
    if args.resume and Path(args.resume).exists():
        print(f"resuming from: {args.resume}")
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        rstate = rck["model"] if "model" in rck else rck
        if any(k.startswith("module.") for k in rstate):
            rstate = {k[len("module."):]: v for k, v in rstate.items()}
        if args.reset_vae:
            n0 = len(rstate); rstate = {k: v for k, v in rstate.items() if not k.startswith("vae.")}
            print(f"  [reset-vae] resume vae.* {n0-len(rstate)}키 strip → --vae-checkpoint({args.vae_checkpoint}) 유지")
        m, u = model.load_state_dict(rstate, strict=False)
        print(f"  model missing={len(m)} unexpected={len(u)}")
        if "optimizer" in rck:
            try:
                optimizer.load_state_dict(rck["optimizer"])
                print("  optimizer state restored")
            except Exception as e:
                print(f"  optimizer restore 실패(무시): {e}")
        # ★ load_state_dict 는 param_groups(lr 포함)를 저장값으로 되돌린다 → --lr 로 강제 재설정
        for g in optimizer.param_groups:
            g["lr"] = args.lr
        print(f"  lr override → {args.lr} (resume 시 저장된 lr 무시)")
        start_step = int(rck.get("step", 0))
        del rck, rstate

    # ── extra-steps: resume step 에 더해서 학습할 step 수로 max_steps 자동계산 ──
    #    (Phase 2 의 '5000 step 더' 를 max-steps 누적값 실수 없이 표현)
    if args.extra_steps is not None:
        args.max_steps = start_step + args.extra_steps
        print(f"  [extra-steps] max_steps = resume_step {start_step} + {args.extra_steps} = {args.max_steps}")
    elif args.resume:
        print(f"  resume step={start_step} (→ max-steps {args.max_steps})")

    # ── dataset (max_steps 확정 후 생성: length 가 정확) ──
    train_fonts = val_fonts = None   # None = 폰트 분할 없음(전체 학습 폰트)
    if args.online_split:
        from korean_split_dataset import make_dataset, split_collate, dump_samples
        # 폰트 단위 train/val 분할: val 폰트는 학습에서 제외 → val 이 '본 적 없는 폰트'가 됨.
        # (--val-fonts-dir 지정 시엔 그 디렉토리가 val, 학습은 전체 폰트 사용)
        if not args.val_fonts_dir and args.val_font_frac > 0:
            from korean_split_dataset import split_train_fonts
            train_fonts, val_fonts = split_train_fonts(args.val_font_frac, seed=args.val_seed)
            print(f"폰트 분할: train {len(train_fonts)} / val {len(val_fonts)} "
                  f"(val-font-frac={args.val_font_frac}, seed={args.val_seed}) "
                  f"| val 폰트: {', '.join(p.stem for p in val_fonts)}")
        # 온라인 무한 생성 = 매 샘플이 새로 렌더됨 (중복 0, 사실상 무한 데이터셋).
        # resume 시 같은 seed 면 처음 본 샘플을 다시 보게 되므로 ckpt 파일명의
        # step 을 seed offset 으로 사용해 항상 새 샘플을 보장.
        seed_off = 0
        if args.resume:
            digits = "".join(c for c in Path(args.resume).stem if c.isdigit())
            seed_off = int(digits) if digits else start_step or 1
        total_len = args.max_steps * args.batch_size * max(1, args.grad_accum)
        if args.english_frac > 0:
            from korean_split_dataset import make_english_dataset
            from torch.utils.data import ConcatDataset
            en_len = int(total_len * args.english_frac)
            ko_len = total_len - en_len
            ko_ds = make_dataset(style_range=tuple(args.style_words), gen_range=tuple(args.gen_words),
                                 length=ko_len, seed=args.seed + seed_off, legacy=args.legacy_transform,
                                 fonts_dir=train_fonts)
            en_ds = make_english_dataset(style_range=tuple(args.style_words), gen_range=tuple(args.gen_words),
                                         length=en_len, seed=args.seed + seed_off + 777,
                                         fonts_dir=args.english_fonts_dir)
            dataset = ConcatDataset([ko_ds, en_ds])
            print(f"combined data: 한글폰트 {ko_len} + 영어폰트 {en_len} "
                  f"(english_frac={args.english_frac}, en fonts={en_ds.renderers_length})")
            if args.save_samples > 0:
                dump_samples(ko_ds, args.save_samples, out_dir / "train_samples",
                             model=model, max_img_len=args.max_img_len)
                dump_samples(en_ds, max(4, args.save_samples // 2), out_dir / "train_samples_en",
                             model=model, max_img_len=args.max_img_len)
        else:
            dataset = make_dataset(style_range=tuple(args.style_words),
                                   gen_range=tuple(args.gen_words),
                                   length=total_len, seed=args.seed + seed_off, legacy=args.legacy_transform,
                                   fonts_dir=train_fonts)
            if args.save_samples > 0:
                dump_samples(dataset, args.save_samples, out_dir / "train_samples",
                             model=model, max_img_len=args.max_img_len)
        if seed_off:
            print(f"online data seed: {args.seed} + offset {seed_off} (resume, 샘플 중복 방지)")
        collate = split_collate
    else:
        dataset = HanDBKoreanDataset(args.lines_json, img_height=64, batch_keys=["style", "same"])
        collate = handb_collate
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate, drop_last=True,
        # 입력 파이프라인 오버랩: 워커가 다음 배치를 미리 렌더(폰트 래스터화는 CPU 병목)
        #   → GPU 가 forward 도는 동안 다음 배치 준비 = GPU 유휴 감소. (numerics 무관, 순수 I/O)
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )

    # ── loss 로그: train_loss.csv (append, resume 시 이어 씀) ──
    loss_csv = out_dir / "train_loss.csv"
    loss_f = open(loss_csv, "a", buffering=1)
    if loss_f.tell() == 0:
        loss_f.write("step,loss,mse,ce,it_s,timestamp\n")
    print(f"loss logged: {loss_csv}")

    # ── truncation 로그: 잘릴 샘플은 학습에 안 쓰고 버림 + 전부 기록 (절대 truncation 금지 정책) ──
    #   max_img_len 예산 초과로 style/gen 이 잘리는 샘플은 조기 EOG 등 학습을 오염시키므로 drop.
    #   버린 샘플의 step/종류/길이/예산/텍스트를 CSV 로 남겨 몇개중 몇개·어떤 예인지 추적.
    trunc_csv = out_dir / "truncated_samples.csv"
    trunc_f = open(trunc_csv, "a", buffering=1)
    if trunc_f.tell() == 0:
        trunc_f.write("step,kind,have_px,budget_px,style_len,gen_len,style_text,gen_text,timestamp\n")
    trunc_seen = 0      # 지금까지 본 총 샘플 수(=drop 후보 분모)
    trunc_dropped = 0   # 그중 잘림으로 버린 수
    print(f"truncation 로그: {trunc_csv} (max_img_len={args.max_img_len} → "
          f"style≤{args.max_img_len//2}px, gen≤예산잔여; 초과 샘플은 drop)")

    def _log_dropped(step, dropped, style_lens, gen_lens):
        nonlocal trunc_dropped
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        for (i, kind, have, budget, st, gt) in dropped:
            trunc_dropped += 1
            esc = lambda s: '"' + str(s).replace('"', "'") + '"'
            trunc_f.write(f"{step},{kind},{have},{budget},{style_lens[i]},{gen_lens[i]},"
                          f"{esc(st)},{esc(gt)},{ts}\n")

    # ── validation 세트(고정): 렌더 후 디스크에 저장 → run 간 재사용 ──
    #    한글(기본: train 에서 분할한 val 폰트 = 학습 미노출) + 영어(english_frac 비율, 학습과 동일).
    val_batches = None
    val_f = None
    if args.online_split and args.val_every > 0:
        from korean_split_dataset import dump_samples
        val_samples, n_val = _load_or_build_val_samples(args, out_dir, val_fonts=val_fonts)
        # 부분배치(bs 미만)는 forward shape 버그 회피 위해 제외
        val_batches = [collate(val_samples[i:i + args.batch_size])
                       for i in range(0, len(val_samples), args.batch_size)]
        val_batches = [b for b in val_batches if len(b["style_img"]) == args.batch_size]
        # 시각화: val 세트 샘플을 학습 샘플과 동일 포맷(montage)으로 저장
        #   model 전달 → '모델이 실제로 받는 style(복원)' 열까지 저장(style-len fix end-to-end 확인)
        dump_samples(val_samples, min(len(val_samples), args.save_samples or 12),
                     out_dir / "val_samples", model=model, max_img_len=args.max_img_len)
        val_csv = out_dir / "val_loss.csv"
        val_f = open(val_csv, "a", buffering=1)
        if val_f.tell() == 0:
            val_f.write("step,val_loss,val_mse,val_ce,timestamp\n")
        en_n = int(round(n_val * args.english_frac)) if args.english_frac > 0 else 0
        if args.val_fonts_dir:
            ko_fonts_desc = Path(args.val_fonts_dir).name
        elif val_fonts:
            ko_fonts_desc = f"train 에서 분할한 {len(val_fonts)}종(학습 제외)"
        else:
            ko_fonts_desc = "train 전체(텍스트만 분리)"
        print(f"val: {len(val_batches)} batches × bs {args.batch_size} "
              f"(한글 {n_val - en_n} + 영어 {en_n}, KO 폰트: "
              f"{ko_fonts_desc}) → {val_csv}, 시각화 → {out_dir/'val_samples'}")

    def _run_val(step):
        if val_batches is None:
            return
        r = _evaluate(model, val_batches, device, args.max_img_len)
        if r is None:
            return
        vl, vm, vc = r
        print(f"  [val] step {step}  val_loss={vl:.4f}  val_mse={vm:.4f}  val_ce={vc:.4f}")
        val_f.write(f"{step},{vl:.6f},{vm:.6f},{vc:.6f},"
                    f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")

    model.train()
    step = start_step
    skipped = 0
    t0 = time.time()
    last_log = t0
    losses_window = []
    mse_window = []
    ce_window = []
    accum = max(1, args.grad_accum)
    micro = 0          # 현재 virtual step 안에서 누적한 micro-batch 수
    acc_loss = 0.0     # virtual step 의 평균 loss 집계용
    acc_mse = 0.0      # 〃 mse (loss 와 동일 스코프로 집계해야 loss=mse+ce 가 맞음)
    acc_ce = 0.0       # 〃 ce
    print(f"virtual batch = {args.batch_size} x {accum} = {args.batch_size * accum}")
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for sample in loader:
            if step >= args.max_steps:
                break

            # ── 잘림 방지: 예산 초과로 style/gen 이 truncation 될 샘플은 배치에서 제거 + 기록 ──
            trunc_seen += len(sample["style_img"])
            orig_slen, orig_glen = sample["style_img_len"], sample["gen_img_len"]
            sample, dropped = filter_no_truncation(sample, args.max_img_len)
            if dropped:
                _log_dropped(step, dropped, orig_slen, orig_glen)
            if len(sample["style_img"]) == 0:
                continue   # 배치가 통째로 잘림 대상 → skip

            # sample = list 들 (handb_collate)
            try:
                model_inputs = model.get_model_inputs(
                    sample["style_img"], sample["gen_img"],
                    sample["style_img_len"], sample["gen_img_len"],
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
                    gen_text=sample["gen_text"],
                )
                losses, pred_latent = out
                loss = losses["loss"]
                mse = losses["mse_loss"]
                ce = losses["ce_loss"]

                (loss / accum).backward()   # gradient accumulation
            except Exception as e:
                # 누적 중 실패 → 부분 누적 폐기 (virtual batch 통째로 drop)
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                acc_loss = 0.0
                skipped += 1
                print(f"[step {step}] forward/backward SKIP ({skipped} total): {e}")
                continue

            acc_loss += loss.item()
            acc_mse += mse.item()
            acc_ce += ce.item()
            micro += 1
            if micro < accum:
                continue                    # 아직 virtual batch 미완성

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses_window.append(acc_loss / accum)
            mse_window.append(acc_mse / accum)
            ce_window.append(acc_ce / accum)
            micro = 0
            acc_loss = 0.0
            acc_mse = 0.0
            acc_ce = 0.0
            step += 1

            if step % args.log_every == 0:
                n_win = min(len(losses_window), args.log_every)
                avg = sum(losses_window[-args.log_every:]) / n_win
                avg_mse = sum(mse_window[-args.log_every:]) / n_win
                avg_ce = sum(ce_window[-args.log_every:]) / n_win
                ips = args.log_every / (time.time() - last_log + 1e-9)
                last_log = time.time()
                drop_rate = trunc_dropped / max(1, trunc_seen)
                print(f"step {step}/{args.max_steps}  loss={avg:.4f}  mse={avg_mse:.4f}"
                      f"  ce={avg_ce:.4f}  {ips:.1f} it/s"
                      f"  truncated_drop={trunc_dropped}/{trunc_seen} ({drop_rate:.1%})")
                loss_f.write(f"{step},{avg:.6f},{avg_mse:.6f},{avg_ce:.6f},"
                             f"{ips:.2f},{datetime.datetime.now().isoformat(timespec='seconds')}\n")

            if args.val_every > 0 and step % args.val_every == 0:
                _run_val(step)

            if args.save_every > 0 and step % args.save_every == 0:
                # jamo_text 는 추론에서 같은 값으로 켜야 하므로 ckpt 에 같이 남긴다
                ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "jamo_text": args.jamo_text}
                p = out_dir / f"checkpoint_step_{step:06d}.pth"
                torch.save(ckpt, p)
                print(f"  saved {p}")

    if args.val_every > 0 and step % args.val_every != 0:
        _run_val(step)   # 종료 시 마지막 val (직전 step에서 이미 했으면 중복 방지)
    final = out_dir / "checkpoint_last.pth"
    torch.save({"model": model.state_dict(), "step": step, "jamo_text": args.jamo_text}, final)
    loss_f.close()
    if val_f is not None:
        val_f.close()
    trunc_f.close()
    dr = trunc_dropped / max(1, trunc_seen)
    print(f"truncation 최종: {trunc_dropped}/{trunc_seen} ({dr:.2%}) drop → {trunc_csv}")
    print(f"done. saved {final}  total {(time.time()-t0)/60:.1f} min")
