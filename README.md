# Eruku Korean Fine-tuning

[Eruku](https://github.com/Blowing-Up-Groundhogs/Eruku) (autoregressive styled handwriting
generation, [arxiv 2510.23240](https://arxiv.org/abs/2510.23240)) 를 **한글 손글씨 라인
이미지 생성**으로 fine-tuning 하는 프로젝트.

- 목표: 한글 + 영어 + 숫자 + 구두점/특수기호 혼합 **문장 line-text 이미지** 생성
- 방식: 공식 영어 pretrained 에서 출발, 온라인 합성 데이터(폰트=writer)로 fine-tune
- 모델: T5-large + ByT5 tokenizer(한글 byte 처리) + 동결 VAE(`emuru_vae`) + 동결 OrigamiNet(영어 OCR, `alpha=1.0` 으로 loss 미사용)

## 빠른 시작

```bash
git clone https://github.com/HERIUN/Eruku_korean_finetuning.git
cd Eruku_korean_finetuning
uv sync                       # 환경 (torch 2.9+cu128, diffusers/transformers/skimage …)
```

저장소의 **모든 기능은 루트의 세 스크립트**로 부른다. 인자 없이 실행하면 사용법이 나온다.

| | | |
|---|---|---|
| **`./train.sh`** | 학습 · 데이터 생성 | `best` `en2ko` `phase1` `phase2` `vae` `htr` `data` `refset` |
| **`./inference.sh`** | 생성 · 시각화 · 릴리즈 | `best` `hf` `show` `matrix` `echo` `refset` `release` |
| **`./eval.sh`** | 정량 평가 · 감사 · 실험 | `best` `cer` `easyocr` `echo` `all` `fonts` `exp` |

```bash
./train.sh                              # 사용법
./train.sh en2ko --lr 1e-5              # 뒤 인자는 그대로 넘어간다
./train.sh en2ko --help                 # 해당 스크립트의 전체 인자

GPU=2 ./eval.sh best                    # 쓸 GPU (기본 0)
DRY=1 ./train.sh best all               # 실행 않고 커맨드만 출력
```

> 세 스크립트는 얇은 디스패처다. 실제 코드는 `train/` `eval/` `infer/` `tools/` 에 있고
> (`./train.sh phase1` = `python train/phase1.py`), 직접 호출해도 똑같이 동작한다.
> 다만 **`.venv/bin/python`** 으로 실행해야 한다 — 시스템 `python` 엔 의존성이 없다(스크립트는 알아서 쓴다).

### 🎯 현재 best 모델 재현

배포본 `finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth` 를 만든 **실제 순서**
(각 run 의 `train_config.yml` 기록 기준)를 7단계로 박아뒀다. 근거·수치는
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) §4·§5.

```bash
./train.sh best status    # 7단계 + 각 산출물이 이미 있는지 표시
./train.sh best 7         # 한 단계만 (번호 또는 이름: short)
./train.sh best all       # 없는 단계만 순서대로 (GPU 수십 시간)
./eval.sh  best           # 재현 검증 — 아래 기대 수치를 같이 출력한다
./inference.sh best       # 눈으로 확인 (생성 그리드 PNG)
```

| # | 단계 | 산출물 | 하는 일 |
|---|---|---|---|
| 1 | `base` | `korean_en2ko_fixed/checkpoint_step_011000.pth` | 영어 pretrained → 한글 (20k step, 채택 s11000) |
| 2 | `htr` | `aux_htr_ko/htr_s20000` | 한글 HTR 리더 (VAE loss + 평가 리더) |
| 3 | `vaedec` | `vae_korean_dec/vae_s15000` | VAE decoder-only 워밍업 |
| 4 | `vaefull` | `vae_korean_full/vae_s15000` | VAE full + HTR 0.3 |
| 5 | `vaemse` | `vae_korean_full_mse/vae_s28000` | VAE full **pure-L1** ← 릴리즈 VAE |
| 6 | `readapt` | `eruku_readapt_fullvae/checkpoint_step_015000.pth` | 새 VAE latent 로 T5 재적응 (`--reset-vae`) |
| 7 | `short` | `eruku_short_fullmse/checkpoint_step_030000.pth` | **단문강조 재적응 = best** |

`./eval.sh best` 가 대조할 기대 수치 (n=100, cfg=1.0, coherent):

| | 전체 | 단문(1~3) | 중문(4~8) | 장문(9~15) |
|---|---|---|---|---|
| 한글 (기록 / **실측 재현**) | 0.127 / **0.121** | 0.286 / **0.269** | 0.049 / **0.047** | 0.059 / **0.060** |
| 영어 (기록 / **실측 재현**) | 0.045 / **0.032** | — / 0.073 | — / 0.015 | — / 0.025 |

실측 재현 = 2026-08-14 에 `./eval.sh best` 를 그대로 돌린 값. 생성은 run-to-run 재현이 안 되므로
±0.01 수준 차이는 정상이고, 판정은 **n≥100 페어링**으로만 한다.

> ⚠️ 3·4단계는 **당시 실제로 거친 warm-start 체인**이라 남겨둔 것이고, 새로 만든다면
> 5단계 레시피(full + pure-L1)를 원본 `emuru_vae` 에서 한 번에 돌리는 쪽이 낫다(§4).
> 원본 run 들이 쓰던 `--val-fonts-dir assets/fonts_korean_unseen` 은 폴더가 정리돼 빠졌다 —
> val_loss 절대값만 달라지고 생성·CER 결과에는 영향 없다.

### 처음부터 학습 (best 재현이 아니라 새로 돌릴 때)

```bash
./train.sh data --n 8 --out /tmp/ksplit   # (선택) 데이터 파이프라인 sanity check

# 논문 2-phase: Phase 1(짧은 어절로 glyph) → Phase 2(긴 줄 일반화)
GPU=0 ./train.sh phase1                   # → finetune_runs/korean_p1/
GPU=0 ./train.sh phase2                   # korean_p1/checkpoint_last.pth 에서 resume, +5000 step
GPU=0 ./train.sh phase1 --batch-size 16 --grad-accum 16 --max-steps 20000   # 인자 덮어쓰기

# (권장 대안) 영어 pretrained → 바로 한글 긴 줄 — Phase 1 생략, 영어능력 보존
GPU=0 ./train.sh en2ko                    # → finetune_runs/korean_en2ko/
```

> 진입점 구조: 공통 로직은 `train/core.py`(전체 인자 `make_parser` + 학습 루프)에 있고,
> 각 Phase 스크립트(`train/phase1.py`/`train/phase2.py`/`train/korean_from_en.py`)가
> `set_defaults` 로 레시피를 덮어씌운 뒤 `train()` 을 호출한다. 개별 인자는 CLI 로 덮어쓰기 가능.

왜 두 단계인가: 논문에서 **Phase 1 (65000 iter × 256 = 16.6M 샘플, 짧은 단어쌍)** 이
byte→glyph 매핑을 만들고, **Phase 2 (5000 iter = 1.28M, 긴 줄)** 는 길이 일반화만 담당.
한글 glyph 는 영어 pretrained 에 없는 **신규 능력**이라 Phase-1 형태(짧은 샘플 = 같은
시간에 더 많은 glyph 감독)로 먼저 가르치는 것이 효율적. 단 from-scratch 는 불필요 —
VAE-latent 디코딩·자기회귀·style 메커니즘은 영어 pretrained 에서 전이됨.

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

### 3-C. VAE 한글 적응 (fidelity 상한 올리기 — 근거는 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) §3)

동결 VAE 의 한글 재구성 한계가 fidelity 하드 상한이라, VAE 자체를 한글에 맞춘다.

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
- `--htr-weight` — OCR 가독성 loss. **기본 0 권장**(켜면 한글 MSE·영어 모두 악화 — EXPERIMENTS.md §4)
- `--pyramid-weight/--pyramid-levels/--pyramid-alphas` — Laplacian band-pass(고주파 우선 가중)
  loss. 기본 0 = 끔. 재현용으로 남긴 실험 경로이며 pure-L1 대비 이득 없음(EXPERIMENTS.md §4)
- `--logvar-weighting` — 원본 LDM 관례의 학습형 관측 노이즈 가중. 기본 off, 우리 세팅선 무익
- `--val-font-frac` — 학습 폰트에서 val 폰트를 떼는 비율(폰트 일반화 측정). VRAM ~20GB(batch 8)

### 무엇이 자동으로 받아지고, 무엇이 안 받아지나

| 언제 | 무엇 | 크기 |
|---|---|---|
| 모델 생성 시 항상 | `google/byt5-small` 토크나이저 + `google-t5/t5-large` **config 만**(가중치 아님 — `T5ForConditionalGeneration(config)` 는 랜덤 초기화) | 작음 |
| 모델 생성 시 항상 | `blowing-up-groundhogs/emuru_vae` VAE 가중치 | ~300MB |
| 학습 시작 (`--resume` 없을 때) | 영어 pretrained `blowing-up-groundhogs/eruku` 의 `pytorch_model.bin` | ~2.9GB |
| `./inference.sh hf` | 발행본 `HERIUN/eruku_korean` + 페어 VAE `HERIUN/emuru_vae_korean` | ~3.2GB |

⚠️ **한글을 아는 T5 본체 가중치(`.pth`)는 자동으로 받지 않는다.** `infer/show.py`·`eval/*` 는
`--ckpt` 로 로컬 파일을 읽고, `finetune_runs/` 는 gitignore 라 새 클론엔 없다. 그래서 새 환경에서는
**`./inference.sh hf`(발행본 사용)** 로 돌리거나 `./train.sh best` 로 직접 만들어야 한다.
`./inference.sh best` 는 로컬 ckpt 가 없으면 알아서 발행본 경로로 넘어간다.
OrigamiNet OCR ckpt 는 불필요 (alpha=1.0 → OCR loss 미사용, 공식 HF 릴리즈에도 OCR 모듈 없음).

### 학습 산출물 (`--out` 디렉토리)

- `train_config.yml` — 실행마다 전체 인자+타임스탬프가 **run 히스토리**로 누적 기록
- `train_loss.csv` — `step,loss,mse,ce,it_s,timestamp` (log-every 마다, resume 시 이어 씀)
- `checkpoint_step_XXXXXX.pth` — model + optimizer + step (resume 용)
- `train_samples/`(+`_en`) — `--save-samples N` 지정 시 학습 데이터 샘플 + montage.
  **모델이 실제로 받는 style 복원 열** 포함(style-len·정규화 fix end-to-end 검증용)
- `val_set.pt` / `val_samples/` — `--val-every>0` 시 **고정 held-out val 세트**(디스크 저장 → run 간 재사용,
  매번 새로 샘플링 안 함). 한글(unseen 폰트)+영어를 `--english-frac` 비율로 섞고, `--val-samples` 로 총량 지정
- `val_loss.csv` — `step,val_loss,val_mse,val_ce,timestamp`
- `truncated_samples.csv` — **절대 truncation 금지 정책**: `max_img_len` 예산 초과로 style/gen 이 잘릴
  샘플은 학습에서 **drop** 하고 여기에 기록(step/종류/길이/텍스트). 몇 개중 몇 개 버렸는지 추적

### 이어 학습 (resume)

```bash
./train.sh phase2 --resume finetune_runs/korean_p2/checkpoint_step_004000.pth --max-steps 200000
```

resume 시 이전 run 의 `train_config.yml` 과 인자가 다르면 `[config WARN]` 으로 알려줍니다
(lr/batch 등 바뀐 채 이어 학습하는 실수 방지).

## 학습 데이터 생성 (온라인 합성, 디스크 0)

학습 데이터는 **디스크에 미리 저장하지 않고 매 step 새로 합성**한다 (`custom_datasets/korean_split.py`
= 원본 repo `OnlineSplitFontSquare` 의 한글 서브클래스). 따라서 모든 샘플이 unique 이고
사실상 무한 데이터셋. 한 샘플 = `(style_img, style_text, gen_img, gen_text)` 4-tuple.

### (1) 폰트 = writer
- **한글 손글씨/디스플레이 폰트 83종** (`assets/fonts_korean_v2/train/` 의 .ttf).
  Eruku 의 "writer(필체)" 개념을 폰트 1종 = writer 1명으로 매핑.
  (초기 손글씨 폰트 + Google Fonts 한글(`tools_fetch_gf_korean.py` 수집)을 병합해 현재 83종)
- 렌더가 깨지는 폰트 3종(NMFClassic, GangwonEduSaeeum, KCCPakKyongni)은 제거됨.
- `fonts_charsets.json` 에 각 폰트의 charset 을 캐시 → 폰트가 못 그리는 글자(두부 □)를
  애초에 샘플하지 않아 tofu 방지.
- **held-out 테스트 폰트 16종** (`assets/fonts_korean_v2/test/`): 학습에 안 쓴 폰트.
  최종 일반화 평가 전용. 학습 중 val 은 기본적으로 train 폰트에서 `--val-font-frac`(기본 0.1)
  비율만큼 결정적으로 분할해 **학습에서 제외**하고 사용 → 본 적 없는 폰트로 val loss 측정.
  (`--val-fonts-dir` 로 특정 디렉토리를 val 로 지정할 수도 있음)
- **영어 폰트 177종** (`assets/fonts_english/`, `--english-fonts-dir` 기본값):
  `--english-frac > 0` 일 때 라틴 폰트 데이터를 섞어 영어 스타일 다양성 보강 (`tools_fetch_gf_english.py` 로 수집).

### (2) 텍스트 = MixedLineSampler 로 합성한 가짜 라인
style 텍스트와 gen(target) 텍스트를 **각각 독립적으로**, 아래 비중으로 토큰을 섞어 1줄 생성
(`custom_datasets/korean_split.py:65`, 원본 논문의 "English + random words" 를 한글로 확장):

| 토큰 종류 | 비중 | 출처 |
|---|---|---|
| 한글 단어 | **0.45** | `assets/corpus/korean_lines.txt` (한 줄당 한 문장, 1.7MB) 에서 어절 추출 |
| 영어 단어 | **0.20** | `assets/corpus/english_words.txt` (466,511 단어) |
| 숫자 | **0.20** | 날짜/시각/금액/전화번호 등 패턴 생성 |
| 랜덤음절 | **0.15** | KS X 1001 **2,350 음절**(전 폰트 교집합)에서 균등추첨한 1~4글자 가짜 단어 |
| + 구두점 0.35 / 특수기호 0.08 확률로 토큰에 부착 | | |

- **랜덤음절(rand)** = 논문 "random words" 대응. 코퍼스 빈도 편향 없이 전 음절을 단어
  맥락에서 노출시켜 희귀 글자까지 학습 (2,000라인 샘플 시 2,347/2,350 음절 등장 확인).
- 어절 수 범위는 Phase 별로 다름: **Phase 1 = style 2~3 / gen 2~3**,
  **Phase 2 = style 1~8 / gen 1~32** (긴 줄). `--style-words` / `--gen-words` 로 조절.

### (3) 렌더 → 증강 → split
1. style/gen 두 텍스트를 **같은 폰트**로 각각 이미지 렌더 (높이 64px).
2. 증강 적용 (`custom_datasets/font_square/font_square_split.py:180-189`):
   `RandomRotation(±3°, p=0.5)`, **`RandomWarping`(TPS 비선형 휨, p=1.0 항상)**,
   `GaussianBlur(p=0.5)`, `GrayscaleDilation(p=0.1)`, `ColorJitter(p=0.5)`,
   종이배경 합성, ink jitter. → 실제 손글씨의 왜곡에 robust 하게 학습.
   (생성 결과가 휘어 보이는 건 이 warp 를 모델이 학습·재현한 정상 동작)
3. 폭 기준으로 잘라 `style_img`(조건) / `gen_img`(target) 로 분리.
- 데이터 sanity check: `uv run python custom_datasets/korean_split.py --n 8 --out /tmp/ksplit`
  → 8개 style/gen pair + montage 저장.

### 원본 repo(`OnlineSplitFontSquare` + `TextSampler`) 대비 변경점

한글판은 원본 온라인 합성 파이프라인을 **서브클래스**(`KoreanSplitFontSquare`)로 재사용하되,
아래 4가지를 바꿨다. 근거는 각 항목의 코드 위치 참조.

| 구분 | 원본 repo | 한글판 (변경) | 이유 |
|---|---|---|---|
| **텍스트 소스** | `TextSampler`: 영어 단어만(nltk 6개 코퍼스 103,411단어), uni/bigram 빈도 가중 1종 sampler로 style·gen 동일 추첨 | `MixedLineSampler`: 한글0.45/영어0.20/숫자0.20/랜덤음절0.15 혼합 + 구두점·특수기호 부착. **폰트 charset 인지**(못 그리는 글자 미추첨 → tofu 방지) | 한글+숫자+기호 혼합 라인이 목표. 빈도편향 없이 전 음절 노출(랜덤음절) |
| **style/gen sampler** | 단일 sampler → style·gen 어절수 범위 동일 | style/gen **독립 sampler**(범위 분리 가능: Phase1 2~3/2~3, Phase2 1~8/1~32) | 조건(style)과 타겟(gen) 길이를 따로 통제 |
| **렌더 방식** | `[style\|gap\|gen]`을 **한 이미지로 합쳐 렌더** → 증강 → 고정 컬럼비로 분할 | style/gen을 **각각 따로 렌더**(경계 없음) | 원본은 회전·warp가 경계를 밀어 style↔gen 침범(잘림/잔상). 분리로 제거 (`custom_datasets/korean_split.py:80-123`) |
| **증강 정책** | 합친 이미지에 일괄: RandomRotation·**RandomWarping(TPS 휨)**·Blur·배경·Dilation·ColorJitter(bright=0.05)·**RandomInvert(p=0.2)** | **RandomWarping 완전 제거**, **RandomInvert 비활성**, RandomRotation은 **style만**(gen 곧게), 배경은 **gen 흰색 강제**(style만 종이배경), ColorJitter **brightness=0**. Blur·Dilation·AlphaChannel(잉크)·Jitter는 **RNG 스냅샷으로 style·gen 동일 실현**(같은 writer 질감 매칭) | 모델 `generate()`가 흰배경·곧은 글자를 출력 → 타겟을 그렇게 학습해야 휨/배경이 전파 안 됨. style은 추론 시 실제 필기라 현실 증강 유지 |

- **폰트(writer)**: 원본 font_square 영어 폰트 → **한글 손글씨/디스플레이 폰트 83종**(`assets/fonts_korean_v2/train/`).
- `--legacy-transform` 플래그로 원본 composite(합쳐 렌더+일괄증강+경계분할) 방식을 그대로 재현 가능(전/후 비교 실험용).
- 원본 대비 **동일하게 유지**한 것: 온라인 무한 생성(디스크 0), 높이 64px, VAE-latent split 구조, `p_uncond` 텍스트 dropout 메커니즘.

### 논문 레시피 매핑
- `--text-dropout 0.05` = p_uncond (style+gen 텍스트 모두 drop → CFG 학습)
- `--style-text-dropout 0.10` = p_drop (Phase 2 전용, style 텍스트만 drop)
- Phase 1: `--style-words 2 3 --gen-words 2 3` / Phase 2: `--style-words 1 8 --gen-words 1 32`
- virtual batch 256: `--grad-accum` × `--batch-size` = 256 (논문 lr 1e-4 는 이 기준)
- 논문 iteration: Phase 1 = 65000 (16.6M 샘플), Phase 2 = 5000 (1.28M 샘플)
- target 패딩 = visual `<EOG>`: 이미 구현되어 있음 — specials 패딩값 1(=EOG) +
  forward 에서 EOG embedding 치환 + CE 로 "gen 종료 후엔 EOG 만" 학습

### 📄 파이프라인 상세 문서 & 알려진 이슈 (`docs/`)
- **`docs/EXPERIMENTS.md`** — 📊 **측정 로그 전문**(§1~§7): 초기 실험 → echo 정량 → VAE 한글
  수용성 → VAE 적응 ablation → Eruku 재적응 → 강건성 → 종성 획 손실. 아래 「결과 요약」의 근거.
- **`docs/MODULE_IO.md`** — 🧩 **학습 모듈 I/O 총정리**: 배치 dict 6개 키(무엇을 넣어야 하나),
  `style_len`/`gen_len` 이 왜 이미지에서 못 얻는 값인지, 텍스트가 T5 encoder 로 들어가는 경로,
  `forward`(loss+pred_latent) / `generate`(라인 이미지) 출력까지 **실제 생성 이미지**로 설명.
  - 재현: `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python docs/_gen_module_io.py`
- **`docs/DATA_GENERATION.md`** — 원본 `OnlineSplitFontSquare` 데이터 생성 전체 흐름(코드 단위).
- **`docs/PIPELINE_STEP_BY_STEP.md`** — 렌더→증강→split 을 단계별 이미지로 시각화.
  긴 텍스트에서 ①회전(`expand=False`)이 양끝을 잘라내고 ②warp 가 split cut 을 글자로 밀어넣는
  문제를 실측·정리(파이프라인은 **2~3어절 짧은 라인 가정** 위에서만 안전).
- **`docs/STYLE_LEN_BUG.md`** — ⚠️ **`get_model_inputs` 단위 버그**: `sl`(픽셀)을
  `style_img_embeds.shape[-1]`(latent=픽셀/8)과 그대로 `min` 비교해 **style/gen 이 실폭의 ~1/8
  (≈글자 1개)로 잘려** 조건으로 들어가던 문제. `* 8` 로 픽셀 환산해 수정(사용률 ~10% → ~100%,
  실측·before/after 이미지 포함). **이 수정 이전 체크포인트는 style 을 거의 못 봤으므로 재학습 권장.**
  - 재현: `PYTHONPATH=. uv run python docs/_verify_style_len.py` (사용률 ~100% 확인)
- **`docs/JONGSEONG_STROKE_LOSS.md`** — ⚠️ **종성(받침) 획 손실 진단·개선 로그 (진행 중)**.
  겹받침 음절정확도가 홑받침보다 22%p 낮은 원인을 VAE/T5 로 분해한 측정과 통제 실험 기록.

## 추론 파라미터: CFG · drop_text · style_text (중요)

생성 품질에 크게 영향을 주는 세 손잡이. 셋은 **서로 직교**하며 역할이 다르다.

### `drop_text` — CFG 켜는 스위치 (추론 시 반드시 `True`)

CFG(classifier-free guidance)는 매 스텝 **텍스트 있는 forward(cond)** 와 **텍스트 없는 forward(uncond)** 를
계산해 `out = uncond + (cond - uncond) * cfg_scale` 로 외삽한다. `drop_text` 는 그 uncond 분기가
무엇을 쓸지 결정한다:

- `drop_text=True` → uncond 가 **학습된 `uncond_embedding`("텍스트 없음" 벡터, 전체 토큰을 이 벡터로 대체)** 사용
  → `cond ≠ uncond` → CFG 작동.
- `drop_text=False` → uncond = cond → `out = cond + 0` → **CFG 무효**(cfg_scale 무시, forward 만 2배 낭비).

학습 때 text-dropout 으로 `uncond_embedding` 을 배워두므로, 추론에서 이걸 켜야(`True`) 그 표현을 활용한다.
CFG 를 안 쓸 때(`cfg_scale=1.0`)는 uncond forward 자체를 안 하므로 `drop_text` 는 무의미하다.

> ⚠️ upstream `blowing-up-groundhogs/eruku` 의 릴리즈 `modeling_eruku.py` 는 `drop_text=False` 가 기본이라
> `cfg_scale=1.25` 를 줘도 CFG 가 **무효**다(영어 in-domain 이라 CFG 없이도 선명해 티가 안 남). 한글 파인튜닝은
> CFG 증폭이 없으면 생성이 흐려지므로(예: 특정 스타일에서 거의 백지), 이 포크·HF 패키지는 `drop_text=True` 로 수정.

### `cfg_scale` — guidance 강도

클수록 텍스트 조건에 강하게 의존(진하고 또렷). 한글은 **1.5~2.0 권장**(원본 기본 1.25보다 높게).

### `style_text` — style 이미지의 전사 (넣으면 정확도 크게↑)

style 참조 이미지의 전사 텍스트. `f"{style_text}<sog>{gen_text}"` 로 gen_text 와 합쳐져 텍스트 조건이 된다.
**빈값("")이어도 동작**하지만(학습 시 `style_text_dropout=0.10` 으로 무전사 강건성 학습), **내용 정확도가
눈에 띄게 떨어진다.** held-out 폰트 CER(n=40, `drop_text=True` 동일 조건) 실측:

| style_text | gen CER↓ | 모델 오류(gap)↓ |
|---|---|---|
| 제공 | **0.197** | 0.063 |
| 빈값 "" | 0.474 | 0.340 |

가능하면 style 이미지의 실제 전사를 넣을 것. (단 위 echo 측정은 전사=목표라 best-case이며, 실사용
전사≠생성텍스트에선 이득이 이보다 작다.) `drop_text` 와는 무관 — 빈값이라고 `drop_text` 를 바꿀 필요 없다.

> ⚠️ **`style_text` 를 채우면 모델이 style 을 '이어서 그린다'(echo) — 크롭은 반드시 SOG 기준으로.**
> `generate()` 는 `style_text` 가 **비어 있지 않으면 SOG 를 디코더에 넣지 않는다**(모델이 스스로 내야 한다).
> 그래서 모델은 SOG 를 낼 때까지 style 텍스트를 계속 그리고, 그 사이 생성된 IMG 토큰마다 latent 열이 쌓인다.
> `style폭+1` 같은 **고정 오프셋으로 자르면 그 echo 가 결과 앞에 그대로 남는다**(gen_text 에 없는 글자가 앞에 붙는
> 증상). 실측: 폰트 6종에서 echo 누수 160~280px, 중앙값 80px·최대 2661px.
> 올바른 경계는 `generate` 가 돌려주는 special 시퀀스의 **첫 SOG 열**이다(`infer.show.gen_arr` 규약).
> `style_text=""` 경로는 SOG 가 강제 주입돼 echo 가 0 이라 이 버그가 안 보인다.

### 요약

| 손잡이 | 역할 | 권장 |
|---|---|---|
| `drop_text` | CFG 켬/끔 | 추론 시 **항상 True** |
| `cfg_scale` | guidance 강도 | 한글 **1.5~2.0** |
| `style_text` | 텍스트 조건에 전사 포함 | **가능하면 채우기** (정확도↑) |

## 추론 / 시각화

```bash
# 0) style ref 세트 — 뷰어들이 이걸 style 로 쓴다 (data/ 는 gitignore 라 새 클론이면 먼저 생성)
./inference.sh refset --per-font-train 5 --per-font-val 0 --no-augment --out data/ref_set_clean

# 1) 가장 빠른 확인: best 모델로 생성 그리드 → _debug/best_show.png
./inference.sh best
./inference.sh best "쓰고 싶은 문장 2024년"
#    로컬 ckpt(finetune_runs/…, gitignore)가 없으면 자동으로 아래 HF 릴리즈 경로로 넘어간다.

# 1') 로컬 ckpt 없이 — 발행된 HF 릴리즈(공개 API)로 생성
./inference.sh hf                                   # HERIUN/eruku_korean + 페어 VAE 자동 다운로드
./inference.sh hf --style-image my_handwriting.png --style-text "전사" --bbox-crop

# 2) 직접 지정: [스타일 ref | 정답 예시 | 생성 cfg=...]
./inference.sh show \
  --ckpt finetune_runs/korean_p2/checkpoint_step_020000.pth \
  --lines-json data/ref_set_clean/train_lines.json \
  --seed-text "오늘 날씨가 좋아서 친구와 공원에서 커피를 마셨다 2024년" \
  --cfgs 1.0 1.25 1.5 1.75 2.0 --n-writers 4 --out _debug/show.png

# 3) 스타일 × 텍스트 매트릭스 / 모델 나란히 비교
./inference.sh matrix --ckpt <ckpt> --out _debug/matrix.png
./inference.sh echo   --out _debug/echo.png

# style 텍스트(ref 전사) 제공 여부:
#   기본        → style ref 의 텍스트를 조건으로 함께 줌
#   --no-style-text → style 이미지만으로 스타일 파악(전사 미제공). 컬럼 라벨에 (no-stext) 표시.
#   둘을 비교하려면 같은 인자로 두 번 실행(--out 만 다르게):
./inference.sh show --ckpt ... --out _debug/show_stext.png
./inference.sh show --ckpt ... --out _debug/show_nostext.png --no-style-text
```

## 평가

```bash
./eval.sh best                        # best 모델 재현 검증 (한글+영어, 기대 수치 대조)
./eval.sh cer  --ckpt <ckpt> --n 100 --coherent            # 한글 주 지표 (HTR CER)
./eval.sh cer  --ckpt <ckpt> --n 100 --coherent --english  # 영어
./eval.sh echo --ckpt <ckpt>          # echo 정량 (MSE/SSIM/LPIPS/FID)
./eval.sh all  --models a=<ckptA> b=<ckptB> --coherent     # 여러 모델 × ko/en 배터리
./eval.sh fonts                       # 폰트 품질 감사 (두부·malformed 글리프)

./eval.sh exp                         # 일회성 실험 스크립트 목록
./eval.sh exp jong_stroke_loss --n-lines 120   # experiments/jong_stroke_loss.py 실행
```

## repo 구조

```
# 진입점 (루트에는 이 3개만)
train.sh                   # 학습·데이터: best|en2ko|phase1|phase2|vae|htr|data|refset
inference.sh               # 생성·시각화·릴리즈: best|hf|show|matrix|echo|refset|release
eval.sh                    # 평가·감사·실험: best|cer|easyocr|echo|all|fonts|exp
# 학습
train/core.py              # 학습 공통 코어 (make_parser 전체 인자 + 학습 루프)
train/phase1.py            # Phase 1 진입점 (짧은 어절, batch128×2, 65000 step)
train/phase2.py            # Phase 2 진입점 (긴 줄, resume, +5000 step)
train/korean_from_en.py    # 영어 pretrained → 한글 직접 (Phase 1 생략)
train/eruku_continous.py   # (원본 repo 스타일 트레이너: wandb/accelerate/webdataset 기반, 참고용)
train/vae_korean.py        # VAE 한글 적응 finetune (--train-part dec|full, htr/pyramid/logvar 항)
train/aux_htr.py           # 한글 HTR 리더 finetune (VAE 의 HTR loss + 평가 리더 겸용)
custom_datasets/korean_aux.py      # HTR 학습용 (라인이미지, 전사) 데이터셋
custom_datasets/korean_alphabet.py  # HTR alphabet (assets/korean_charset.json 기반)
tools/fetch_aux.py         # HTR 학습용 aux 데이터 수집
# 데이터
custom_datasets/korean_split.py    # 온라인 한글 split 데이터셋 (+ CLI 샘플 덤프, --show-model-input)
custom_datasets/korean_fontset.py      # 오프라인 라인 생성기 + MixedLineSampler/build_pools (공용)
assets/download_script/tools_fetch_gf.py  # Google Fonts 수집기 (--lang ko|en, 한/영 통합)
assets/font_view/render_overview.py       # 폰트 오버뷰 montage (샘플 렌더 + 글자수 분류)
tools/font_audit.py        # 폰트 품질 감사: 미지원(두부)·malformed 글리프 스캔 (코퍼스 기준)
# 추론 / 평가  (infer/show.py 가 공유 라이브러리: load_model/gen_arr/render_in_font …)
infer/release.py           # 발행된 HF 릴리즈(공개 API)로 생성 — 로컬 ckpt 불필요
infer/show.py              # 자기설명적 결과 뷰어 [스타일ref|정답|생성 cfg…] (--exclude-writers 로 폰트 제외)
infer/matrix.py            # 스타일×텍스트 매트릭스 뷰어 (1 style → 여러 gen_text)
infer/echo_compare.py      # echo(style==gen) 모델 나란히 비교 뷰어
eval/echo_metrics.py       # echo 정량 (MSE/SSIM/LPIPS/FID)
eval/cer.py                # echo CER (easyocr 기반 — 한글은 saturated, 교차검증용)
eval/htr_cer.py            # 생성 CER (한글 HTR 리더, 렌더바닥 ~0.00) ← 한글 주 지표
eval/all.py                # 통합 평가 배터리 (eval/cer.py + eval/echo_metrics.py 오케스트레이션)
tools/hf_upload.py         # HF 릴리즈 (ckpt→릴리즈 클래스 변환 + VAE/모델 repo 업로드)
# 일회성 실험/검증 스크립트 (결론은 docs/EXPERIMENTS.md 에 기록됨. experiments/common.py 가
# 저장소 루트를 sys.path 에 넣어주므로 어느 cwd 에서든 `python experiments/xxx.py` 로 실행)
experiments/vae_recon.py         # VAE roundtrip 복원 비교 (고정 probe: ko/en/극단폰트)
experiments/vae_hw_recon.py      # 실제 손글씨 복원 비교 (일반화)
experiments/vae_compare.py       # VAE 여러 개 나란히 비교
experiments/vae_height_sweep.py  # 입력 height 스윕 — h>64 이득을 self/native 두 기준으로 측정
experiments/vae_norm_check.py    # [-1,1] vs [0,1] 정규화 검증 (복원 + Eruku 생성까지)
experiments/vae_prep_sweep.py    # 입력 전처리 스윕 (self MSE/SSIM + HTR CER damage)
experiments/vae_prep_minrec.py   # 최소 vs 권장 전처리 — 공통 캔버스 기준 비교
experiments/gen_compare.py       # Eruku 생성 전/후 montage (같은 VAE → 순수 T5 개선분)
experiments/gen_heldout.py       # held-out 폰트(custom_hw_fonts) 생성 montage
experiments/mirror_compare.py    # 실제 손글씨 style 미러링 + ink bbox 크롭 검증
experiments/jong_stroke_loss.py  # 종성(받침) 획 손실 — VAE roundtrip 을 음절 창 단위로 픽셀 정밀 측정
experiments/jong_gen_loss.py     # 종성 획 손실 — 생성물을 HTR 로 읽어 자모 단위(삭제/오치환) 측정
experiments/jong_hypotheses.py   # 위 두 덤프 재분석 — 획 수(H1)/학습 빈도(H2) 가설 검증(GPU 불필요)
# 모델 / 원본 코드
models/eruku.py    # Emuru 모델 (forward/generate) — style-len 단위버그 수정됨
custom_datasets/           # 원본 repo 데이터 코드 (font_square 렌더/증강, tps)
models/                    # OrigamiNet 등
docs/                      # 파이프라인 상세/버그 분석 문서 + 재현 스크립트·이미지
assets/
  fonts_korean_v2/train/   # 한글 손글씨/디스플레이 폰트 83개 + fonts_charsets.json
  fonts_korean_v2/test/    # held-out 테스트 폰트 16개 (최종 일반화 평가 전용)
  fonts_english/           # 영어(라틴) 폰트 177개 (--english-fonts-dir 기본값)
  fonts_label/             # NanumGothic (뷰어 라벨용)
  backgrounds/             # 종이 배경
  corpus/                  # korean_lines.txt(한 줄당 한 문장) / chars.txt / english_words.txt
  font_view/               # 폰트 오버뷰 스크립트 + 폴더별 overview_*.png
  custom_hw_fonts/         # AI 손글씨 폰트 4종 — 학습 미사용, held-out 평가 전용
  korean_charset.json      # HTR alphabet 문자 집합(2509자)
custom_datasets/hw_line_sample/  # 실제 손글씨 라인 20장 + label.txt (미러링/복원 평가용)
```

### 폰트 오버뷰 (어떤 글씨체인지 한눈에 보기)

```bash
uv run python assets/font_view/render_overview.py    # assets 폰트 폴더 전체 → assets/font_view/overview_*.png
uv run python assets/font_view/render_overview.py --dirs assets/fonts_korean_v2/test   # 특정 폴더만
```

폴더별 montage PNG 생성: 행마다 [파일이름 + 지원 글자수 분류(전체/한글/한자/영문/숫자/기호/기타) |
샘플 1줄 렌더(한글+영어 대/소문자+숫자+특수기호)]. 폰트 추가/교체 후 다시 실행하면 갱신됨.

## 결과 요약

측정 로그 전문(표·재현 커맨드·중간 실험)은 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) 에 있다.
여기엔 **지금 코드·체크포인트에 반영된 결론만** 남긴다.

| 주제 | 결론 | 상세 |
|---|---|---|
| 학습률 | 영어 pretrained init 에서 **lr 5e-5 발산**(virtual batch 256 기준 1e-4). lr 은 virtual batch 에 맞출 것 | [§1](docs/EXPERIMENTS.md) |
| 한글 fidelity 상한 | 동결 VAE 의 **한글 재구성이 하드상한** — 한글 오차가 영어의 **~12배**. 학습을 늘려도 안 넘는다 | [§3](docs/EXPERIMENTS.md) |
| VAE 한글 적응 | `--train-part full` + `--htr-weight 0`(pure-L1) 이 최선. roundtrip MSE **−83%**. pyramid/HTR/logvar 항은 이득 없음 | [§4](docs/EXPERIMENTS.md) |
| Eruku 재적응 | full VAE 는 latent 가 이동 → T5 재적응 필수. 단문강조 재적응으로 한글 CER **0.211 → 0.127**, 단문 **0.54 → 0.29** | [§5](docs/EXPERIMENTS.md) |
| VAE 강건성 | 실제 스캔 손글씨는 **여백이 latent std 를 6~29배**로 튀겨 runaway → ink bbox 크롭 필요(재학습 불필요) | [VAE_ROBUSTNESS.md](docs/VAE_ROBUSTNESS.md) |
| 종성 획 손실 | 겹받침 음절정확도가 홑받침보다 **22%p** 낮다. 원인은 받침 개수가 아니라 **글리프 밀도**(조사 진행 중) | [JONGSEONG_STROKE_LOSS.md](docs/JONGSEONG_STROKE_LOSS.md) |
| 지표 선택 | 픽셀 지표(MSE/SSIM)는 글자 정렬오차에 민감·언어간 비교 부적절. **한글 주 지표는 HTR CER**(`eval/htr_cer.py`) | [§2](docs/EXPERIMENTS.md) |

**현재 best 체크포인트**: `finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth`
(한글 전체 CER 0.127 / 단문 0.286 / 장문 0.059, 영어 0.045 — n=100, cfg=1.0, coherent).
val_mse 는 s20000 이 더 낮지만 생성 CER 은 s30000 이 낫다 — **ckpt 선택은 val_mse 가 아니라 CER 로**.

> 실행 환경 주의: 이 repo 는 **`.venv/bin/python`** 으로 실행한다(torch 2.9+cu128,
> diffusers/skimage/transformers 포함). 시스템 `python`(miniconda base)에는 numpy 조차 없다.

## HuggingFace 릴리즈 (2026-07)

최고 조합을 **repo 2개**로 발행한다. 업로드/변환은 `tools/hf_upload.py` 로 재현한다.

| repo | 내용 | 왜 분리하나 |
|---|---|---|
| [`HERIUN/eruku_korean`](https://huggingface.co/HERIUN/eruku_korean) | T5 + proj + 특수토큰 (`model.safetensors`) + remote code(`modeling_eruku.py`) | 상위 진입점 |
| [`HERIUN/emuru_vae_korean`](https://huggingface.co/HERIUN/emuru_vae_korean) | 한글 적응 VAE (diffusers `AutoencoderKL`) | `modeling_eruku.py` 가 `AutoencoderKL.from_pretrained(config.vae_name_or_path)` 로 **VAE 가중치를 그 repo 에서 실제로 내려받기** 때문 |

**T5 는 별도 업로드가 필요 없다** — `T5Config.from_pretrained(config.t5_name_or_path)` 로 구조만
받고 가중치는 상위 repo 의 weight 파일에서 전부 덮인다. 반면 VAE 는 가중치를 원격에서 가져오므로,
`full` 로 학습해 latent 가 이동한 우리 VAE 를 **config 가 가리키게** 해야 조합이 맞는다
(원본 `blowing-up-groundhogs/emuru_vae` 를 가리킨 채 T5 만 갈면 생성이 무너진다).

```bash
# 0) 카드용 before/after figure (영문 라벨, /tmp/relfig 에 4장)
#    - VAE 복원: 원본 emuru_vae vs 릴리즈 VAE (한글/영어)
for L in ko en; do CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/vae_recon.py --lang $L --label-lang en \
  --models "original=blowing-up-groundhogs/emuru_vae" "ours=finetune_runs/vae_korean_full_mse/vae_s28000" \
  --out /tmp/relfig/recon_$L.png; done
#    - 생성: 원본 영어 pretrained(원본 VAE) vs 릴리즈(한글 VAE + 재적응 T5)
PRE=$(ls ~/.cache/huggingface/hub/models--blowing-up-groundhogs--eruku/snapshots/*/pytorch_model.bin | head -1)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/gen_compare.py --ckpt-before "$PRE" \
  --ckpt-after finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth \
  --label-before "BEFORE: original pretrained (English, original VAE)" \
  --label-after "AFTER: this release (Korean VAE + re-adapted T5)" \
  --col-labels "style reference (input)" "target (rendered in that font)" \
  --row-label-fmt "style: {font}" "target: {text}" \
  --cats ko_release en_release --n-rows 4 --out-dir /tmp/relfig

# 1) VAE 먼저 (상위 repo 로드가 이걸 요구).  --figures-dir 주면 samples/ 로 올리고 카드에 삽입
.venv/bin/python tools/hf_upload.py vae --figures-dir /tmp/relfig --push
# 2) 상위 모델: 변환 → 키 정합성 assert(missing/unexpected 0) → 생성 parity → 업로드
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/hf_upload.py model --verify-cer   # dry-run
.venv/bin/python tools/hf_upload.py model --figures-dir /tmp/relfig --push
```

- 변환 시 로컬 학습 클래스에만 있는 `t5_to_ocr.*`(alpha=1.0 이라 미사용) 1키를 strip 하고,
  `vae.*` 252키는 그대로 남긴다(config 의 VAE repo 와 동일 가중치 = 이중 안전판)
- `save_pretrained` 가 `auto_map` 의 `AutoConfig` 항목을 떨어뜨려 `trust_remote_code` 로드가
  깨지는 문제 → 원본 값으로 보완한다
- ⚠️ **릴리즈 클래스와 학습 클래스는 style prefix 규약이 다르다.** 릴리즈
  `get_model_inputs` 는 style latent 뒤에 SOG placeholder 열(ones) 1개를 붙이고,
  `generate` 가 반환 전에 style prefix 를 **내부에서 잘라낸다**. 로컬은 안 붙이고 안 자르며
  `infer.show.gen_arr` 가 special 시퀀스의 첫 SOG 로 자른다. 둘을 섞어(릴리즈 출력에 gen_arr
  크롭을 겹쳐) 쓰면 이중 크롭으로 생성물이 날아간다 — 릴리즈 경로는 `generate_handwriting`
  하나로만 쓸 것. 검증은 `--verify-cer`(공개 API 그대로 CER 측정)로 한다
- ⚠️ **(2026-08 수정) 그 내부 크롭이 `style폭+1` 고정 오프셋이라 style echo 가 남았다.**
  `style_text` 를 주면 모델이 SOG 를 낼 때까지 style 을 이어 그리는데(위 「`style_text`」 경고),
  고정 오프셋은 그 사이 열들을 못 자른다 → 생성물 앞에 gen_text 에 없는 글자가 붙는다.
  `tools.hf_upload.patch_modeling` 이 이 줄을 **special 시퀀스의 첫 SOG 기준**으로 바꾼다(로컬과 동일 규약).
  릴리즈 경로 CER 실측(n=100, 같은 텍스트·폰트로 페어링, cfg=1.0):

  | 구간 | 패치 전 | 패치 후 |
  |---|---|---|
  | 전체 | 0.494 | **0.163** |
  | 단문(1~3) | 1.253 | **0.391** |
  | 중문(4~8) | 0.159 | **0.049** |
  | 장문(9+) | 0.098 | **0.066** |

  표본별 66승/29무/5패이고, 진 5개는 전부 echo 가 없던(폭 차이 ≤40px) 표본이라 패치가 아니라 실행 편차다.
  `style_text=""` 경로는 크롭 인덱스가 완전히 동일 → 회귀 없음.
  ✅ **2026-08-14 발행 완료** ([commit `0e3ce43`](https://huggingface.co/HERIUN/eruku_korean/commit/0e3ce430fde7372f71639fce8369846666c1d167)).
  발행 전 검증: 키 정합성 missing/unexpected 0, 공개 API 실측 CER 0.052(n=24, GT바닥 0.003),
  bbox 크롭 smoke test 통과. 발행 후 `./inference.sh hf` 재실행으로 echo 소멸을 눈으로 확인했다
  (패치 전에는 생성물이 style 잔재 `산이:`·`권한:` 로 시작했다).
- ⚠️ **생성은 run-to-run 재현이 안 된다.** 같은 모델·같은 입력 2회에 픽셀 maxdiff 221, `--verify-cer`
  (n=24) 3회에 CER 0.035 / 0.031 / 0.074. 비결정적 GPU 커널이 argmax 를 뒤집으면 그 뒤 생성이 통째로
  갈라지기 때문. **n=24 는 게이트용이지 보고용 수치가 아니다** — 비교는 n≥100 페어링으로 할 것
- 업로드하는 `modeling_eruku.py` 에는 **ink bbox 크롭**(`generate_handwriting(bbox_crop=True)`)을
  추가했다. 실제 스캔 손글씨의 여백이 latent std 를 폰트의 6~29배로 튀겨 runaway/blank 를
  만드는 문제(→ [`docs/VAE_ROBUSTNESS.md`](docs/VAE_ROBUSTNESS.md) ②)의 추론측 해결책

```python
from transformers import AutoModel
from PIL import Image
m = AutoModel.from_pretrained("HERIUN/eruku_korean", trust_remote_code=True).eval().cuda()
img = m.generate_handwriting(style_image=Image.open("style_line.png"),
                             style_text="조용히 책을 읽었다.",   # style 전사(주면 정확도↑)
                             gen_text="한국어 손글씨 생성", cfg_scale=1.5)
```

## 요구 사양

- CUDA GPU ≥ 24 GB (batch 2, max-img-len 2048 기준; T5-large 705M trainable).
  VAE 적응(`train/vae_korean.py` batch 8 × accum 4)은 ~20 GB, 한글 HTR(`train/aux_htr.py`)은 ~10 GB
- 디스크: repo ~250 MB + pretrained 2.9 GB(자동 다운로드) + 학습 ckpt 개당 ~8 GB
- TPS C++ 백엔드는 선택 (`custom_datasets/font_square/tps/build.sh`, `uv sync --extra tps` 후) — 없으면 NumPy fallback 으로 동작
