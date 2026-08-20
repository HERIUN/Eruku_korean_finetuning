# 새로 학습하기 (레시피 · 하이퍼파라미터)

배포본을 그대로 재현하려면 [`REPRODUCE.md`](REPRODUCE.md) 를 볼 것. 이 문서는 **새로 돌릴 때** 의
레시피다. 데이터가 어떻게 합성되는지는 [`DATA_GENERATION.md`](DATA_GENERATION.md) §8.

> 진입점 구조: 공통 로직은 `train/core.py`(전체 인자 `make_parser` + 학습 루프)에 있고,
> 각 Phase 스크립트(`train/phase1.py`/`train/phase2.py`/`train/korean_from_en.py`)가
> `set_defaults` 로 레시피를 덮어씌운 뒤 `train()` 을 호출한다. 개별 인자는 CLI 로 덮어쓰기 가능.

## 0. 무엇이 자동으로 받아지나

| 언제 | 무엇 | 크기 |
|---|---|---|
| 모델 생성 시 항상 | `google/byt5-small` 토크나이저 + `google-t5/t5-large` **config 만**(가중치 아님 — `T5ForConditionalGeneration(config)` 는 랜덤 초기화) | 작음 |
| 모델 생성 시 항상 | `blowing-up-groundhogs/emuru_vae` VAE 가중치 | ~300MB |
| 학습 시작 (`--resume` 없을 때) | 영어 pretrained `blowing-up-groundhogs/eruku` 의 `pytorch_model.bin` | ~2.9GB |
| `./inference.sh hf` | 발행본 `HERIUN/eruku_korean` + 페어 VAE `HERIUN/emuru_vae_korean` | ~3.2GB |

안 받아지는 것: 학습된 T5 본체 `.pth`(`finetune_runs/` 는 gitignore) — 발행본으로 대체하거나
직접 학습해야 한다(→ [README 「학습 없이 쓰기」](../README.md#학습-없이-쓰기-발행본)).
OrigamiNet OCR ckpt 는 불필요하다 (`alpha=1.0` 이라 loss 미사용, 공식 HF 릴리즈에도 OCR 모듈 없음).

## 1. 처음부터 학습

```bash
./train.sh data --n 8 --out /tmp/ksplit   # (선택) 데이터 파이프라인 sanity check

# 논문 2-phase: Phase 1(짧은 어절로 glyph) → Phase 2(긴 줄 일반화)
GPU=0 ./train.sh phase1                   # → finetune_runs/korean_p1/
GPU=0 ./train.sh phase2                   # korean_p1/checkpoint_last.pth 에서 resume, +5000 step
GPU=0 ./train.sh phase1 --batch-size 16 --grad-accum 16 --max-steps 20000   # 인자 덮어쓰기

# (권장 대안) 영어 pretrained → 바로 한글 긴 줄 — Phase 1 생략, 영어능력 보존
GPU=0 ./train.sh en2ko                    # → finetune_runs/korean_en2ko/
```

### 왜 두 단계인가

논문에서 **Phase 1 (65000 iter × 256 = 16.6M 샘플, 짧은 단어쌍)** 이 byte→glyph 매핑을 만들고,
**Phase 2 (5000 iter = 1.28M, 긴 줄)** 는 길이 일반화만 담당한다. 한글 glyph 는 영어 pretrained 에
없는 **신규 능력**이라 Phase-1 형태(짧은 샘플 = 같은 시간에 더 많은 glyph 감독)로 먼저 가르치는
것이 효율적이다. 단 from-scratch 는 불필요 — VAE-latent 디코딩·자기회귀·style 메커니즘은 영어
pretrained 에서 전이된다. (실제 배포본은 Phase 1 을 생략했다 → [`REPRODUCE.md`](REPRODUCE.md))

### step · 노출량 · VRAM · lr

- `--max-steps` / `--extra-steps` / `--save-every` / `--log-every` 는 **virtual step**(optimizer 업데이트) 기준.
  Phase 2 는 `--extra-steps N` 으로 `max-steps = resume_step + N` 을 자동계산 → 누적값 실수 방지
- 노출량(기본 레시피): Phase 1 65000 step × 256 = **16.6M 샘플**, Phase 2 5000 step × 256 = **1.28M**.
  논문은 10M 고정 데이터셋을 재사용하지만 여기선 온라인 무한 생성이라 **모든 샘플 unique**
  (resume 시에도 seed offset 으로 중복 방지). 노출량은 `--max-steps` 로 조절
- VRAM: `train/korean_from_en.py`(batch 16 × accum 16, max-img-len 8192) 측정 **~35GB**.
  Phase 1 batch 128 은 훨씬 큰 VRAM(멀티 GPU) 필요 — 단일 24GB 면 `--batch-size 16 --grad-accum 16`
  으로 낮춰 virtual 256 을 유지
- lr 은 **virtual batch 에 맞춰야** 함 (논문 lr 1e-4 = virtual 256 기준). virtual batch 를 줄이면 lr 도
  낮출 것 (batch 2 + lr 5e-5 발산 확인). 빠른 sanity: `--grad-accum 1 --lr 1e-5`

## 2. VAE 한글 적응 (fidelity 상한 올리기)

동결 VAE 의 한글 재구성 한계가 fidelity 하드 상한이라, VAE 자체를 한글에 맞춘다.
근거는 [`EXPERIMENTS.md`](EXPERIMENTS.md) §3.

```bash
# (1) 평가/loss 용 한글 HTR 리더 (easyocr 은 한글 GT 조차 CER 0.236 으로 saturated)
./train.sh htr --out finetune_runs/aux_htr_ko

# (2) VAE finetune — 권장 레시피: full(enc+dec) + HTR off(pure-L1)
./train.sh vae --htr-checkpoint finetune_runs/aux_htr_ko/htr_s20000 \
  --train-part full --htr-weight 0 --kl-weight 1e-6 \
  --batch-size 8 --grad-accum 4 --lr 1e-4 --max-steps 30000 \
  --out finetune_runs/vae_korean_full     # → vae_sXXXXX/ (diffusers AutoencoderKL 폴더)

# (3) 교체 평가 (decoder-only 는 재학습 없이 바로)
./eval.sh cer --ckpt <eruku ckpt> \
  --vae-checkpoint finetune_runs/vae_korean_full/vae_s30000 --n 100 --coherent

# (4) full 은 latent 가 이동하므로 Eruku 재적응 필수 (--reset-vae 로 ckpt 의 옛 vae.* strip)
./train.sh en2ko \
  --resume <한글 ckpt> --vae-checkpoint finetune_runs/vae_korean_full/vae_s30000 --reset-vae \
  --out finetune_runs/eruku_readapt --extra-steps 15000
```

- `--train-part dec|full` — `dec` = latent 불변(교체만으로 사용) / `full` = 상한↑ 대신 재적응 필요
- `--htr-weight` — OCR 가독성 loss. **기본 0 권장**(켜면 한글 MSE·영어 모두 악화 — [`EXPERIMENTS.md`](EXPERIMENTS.md) §4)
- `--pyramid-weight/--pyramid-levels/--pyramid-alphas` — Laplacian band-pass(고주파 우선 가중)
  loss. 기본 0 = 끔. 재현용으로 남긴 실험 경로이며 pure-L1 대비 이득 없음([`EXPERIMENTS.md`](EXPERIMENTS.md) §4)
- `--logvar-weighting` — 원본 LDM 관례의 학습형 관측 노이즈 가중. 기본 off, 우리 세팅선 무익
- `--val-font-frac` — 학습 폰트에서 val 폰트를 떼는 비율(폰트 일반화 측정). VRAM ~20GB(batch 8)

전체 인자 목록은 `train/core.py:make_parser()`, 레시피 기본값은 `train/phase1.py` /
`train/phase2.py` / `train/korean_from_en.py` 의 `set_defaults`.
