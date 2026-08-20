# 현재 best 모델 재현 (7단계)

배포본 `finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth` 를 만든 **실제 순서**
(각 run 의 `train_config.yml` 기록 기준)를 7단계로 박아뒀다. 근거·수치는
[`EXPERIMENTS.md`](EXPERIMENTS.md) §4·§5.

```bash
./train.sh best status    # 7단계 + 각 산출물이 이미 있는지 표시
./train.sh best 7         # 한 단계만 (번호 또는 이름: short)
./train.sh best all       # 없는 단계만 순서대로 (GPU 수십 시간)
./eval.sh  best           # 재현 검증 — 아래 기대 수치를 같이 출력한다
./inference.sh best       # 눈으로 확인 (생성 그리드 PNG)
```

## 단계

| # | 단계 | 산출물 | 하는 일 |
|---|---|---|---|
| 1 | `base` | `korean_en2ko_fixed/checkpoint_step_011000.pth` | 영어 pretrained → 한글. **Phase 1 생략** — `korean_from_en.py` 로 긴 줄(style 1~8 / gen 1~32)만 학습 (20k step, 채택 s11000) |
| 2 | `htr` | `aux_htr_ko/htr_s20000` | 한글 HTR 리더 (VAE loss + 평가 리더) |
| 3 | `vaedec` | `vae_korean_dec/vae_s15000` | VAE decoder-only 워밍업 |
| 4 | `vaefull` | `vae_korean_full/vae_s15000` | VAE full + HTR 0.3 |
| 5 | `vaemse` | `vae_korean_full_mse/vae_s28000` | VAE full **pure-L1** ← 릴리즈 VAE |
| 6 | `readapt` | `eruku_readapt_fullvae/checkpoint_step_015000.pth` | 새 VAE latent 로 T5 재적응 (`--reset-vae`) |
| 7 | `short` | `eruku_short_fullmse/checkpoint_step_030000.pth` | **단문강조 재적응 = best** |

## 기대 수치

`./eval.sh best` 가 대조할 값 (n=100, cfg=1.0, coherent):

| | 전체 | 단문(1~3) | 중문(4~8) | 장문(9~15) |
|---|---|---|---|---|
| 한글 (기록 / **실측 재현**) | 0.127 / **0.121** | 0.286 / **0.269** | 0.049 / **0.047** | 0.059 / **0.060** |
| 영어 (기록 / **실측 재현**) | 0.045 / **0.032** | — / 0.073 | — / 0.015 | — / 0.025 |

실측 재현 = 2026-08-14 에 `./eval.sh best` 를 그대로 돌린 값. 생성은 run-to-run 재현이 안 되므로
±0.01 수준 차이는 정상이고, 판정은 **n≥100 페어링**으로만 한다.

> 로컬 ckpt 가 없으면 `./eval.sh best` 는 HF 발행본으로 자동 폴백한다 — 가중치가
> bit-identical 이라 재현 검증에 그대로 쓸 수 있다 (→ [`RELEASE.md`](RELEASE.md)).

## 주의

ℹ️ 1단계는 **`train/phase2.py` 가 아니다.** phase2 는 Phase 1 ckpt resume 전제(lr 1e-4)라 영어
pretrained 에서 바로 못 쓴다. 어절 범위·dropout·virtual batch 는 phase2 와 같고, 시작점(영어
pretrained)·lr(5e-5)·영어 15% 혼합·val 로깅만 다른 `korean_from_en.py` 를 쓴다.

⚠️ 3·4단계는 **당시 실제로 거친 warm-start 체인**이라 남겨둔 것이고, 새로 만든다면
5단계 레시피(full + pure-L1)를 원본 `emuru_vae` 에서 한 번에 돌리는 쪽이 낫다
([`EXPERIMENTS.md`](EXPERIMENTS.md) §4).
원본 run 들이 쓰던 `--val-fonts-dir assets/fonts_korean_unseen` 은 폴더가 정리돼 빠졌다 —
val_loss 절대값만 달라지고 생성·CER 결과에는 영향 없다.
