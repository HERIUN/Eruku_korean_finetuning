# Eruku Korean Fine-tuning

[Eruku](https://github.com/Blowing-Up-Groundhogs/Eruku) (autoregressive styled handwriting
generation, [arxiv 2510.23240](https://arxiv.org/abs/2510.23240)) 를 **한글 손글씨 라인
이미지 생성**으로 fine-tuning 하는 프로젝트.

- 목표: 한글 + 영어 + 숫자 + 구두점/특수기호 혼합 **문장 line-text 이미지** 생성
- 방식: 공식 영어 pretrained 에서 출발, 온라인 합성 데이터(폰트=writer)로 fine-tune
- 모델: T5-large + ByT5 tokenizer(한글 byte 처리) + 동결 VAE(`emuru_vae`) + 동결 OrigamiNet(영어 OCR, `alpha=1.0` 으로 loss 미사용)

## 빠른 시작 (새 머신)

```bash
git clone https://github.com/HERIUN/Eruku_korean_finetuning.git
cd Eruku_korean_finetuning

# 1. 환경 (uv)
uv sync

# 2. (선택) 데이터 파이프라인 sanity check — 학습 샘플 8개 저장
uv run python korean_split_dataset.py --n 8 --style 1 8 --gen 1 32 --out /tmp/ksplit

# 3-A. 한글 Phase 1 — 한글 glyph 학습 (논문 레시피 내장: 짧은 2~3어절,
#      virtual 256 = batch 128 × accum 2, lr 1e-4, 65000 step, num-workers 16)
CUDA_VISIBLE_DEVICES=0 uv run python train_phase1.py
#   → finetune_runs/korean_p1/  (정상완료 시 checkpoint_last.pth → Phase 2 가 resume)
#   개별 인자 덮어쓰기 가능:  uv run python train_phase1.py --batch-size 16 --grad-accum 16 --max-steps 20000

# 3-B. 한글 Phase 2 — 긴 줄 일반화 (Phase 1 ckpt 에서 resume, +5000 step)
#      논문 그대로: batch 2 × accum 128 = virtual 256, style 1~8 / gen 1~32
CUDA_VISIBLE_DEVICES=0 uv run python train_phase2.py
#   기본으로 korean_p1/checkpoint_last.pth 에서 resume, --extra-steps 5000 (= resume_step+5000 자동)
#   다른 ckpt: uv run python train_phase2.py --resume finetune_runs/korean_p1/checkpoint_step_065000.pth

# (대안) 영어 pretrained → 바로 한글 긴 줄 — Phase 1 생략, 영어능력 보존
#        lr 5e-5, max-img-len 8192, batch 16 × accum 16, english-frac 0.15, held-out val 로깅
CUDA_VISIBLE_DEVICES=0 uv run python train_korean_from_en.py   # → finetune_runs/korean_en2ko/
```

> 진입점 구조: 공통 로직은 `train_core.py`(전체 인자 `make_parser` + 학습 루프)에 있고,
> 각 Phase 스크립트(`train_phase1.py`/`train_phase2.py`/`train_korean_from_en.py`)가
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
- VRAM: `train_korean_from_en.py`(batch 16 × accum 16, max-img-len 8192) 측정 **~35GB**.
  Phase 1 batch 128 은 훨씬 큰 VRAM(멀티 GPU) 필요 — 단일 24GB 면 `--batch-size 16 --grad-accum 16`
  으로 낮춰 virtual 256 을 유지
- lr 은 **virtual batch 에 맞춰야** 함 (논문 lr 1e-4 = virtual 256 기준). virtual batch 를 줄이면 lr 도
  낮출 것 (batch 2 + lr 5e-5 발산 확인). 빠른 sanity: `--grad-accum 1 --lr 1e-5`

첫 실행 시 HuggingFace 에서 `google-t5/t5-large`(config), `google/byt5-small`(tokenizer),
`blowing-up-groundhogs/emuru_vae`, 그리고 **영어 pretrained weight**
(`blowing-up-groundhogs/eruku` 의 `pytorch_model.bin`, ~2.9GB) 를 자동 다운로드하므로
인터넷이 필요합니다. OrigamiNet OCR ckpt 는 불필요 (alpha=1.0 → OCR loss 미사용,
공식 HF 릴리즈에도 OCR 모듈 없음).

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
uv run python train_phase2.py \
  --resume finetune_runs/korean_p2/checkpoint_step_004000.pth --max-steps 200000
```

resume 시 이전 run 의 `train_config.yml` 과 인자가 다르면 `[config WARN]` 으로 알려줍니다
(lr/batch 등 바뀐 채 이어 학습하는 실수 방지).

## 학습 데이터 생성 (온라인 합성, 디스크 0)

학습 데이터는 **디스크에 미리 저장하지 않고 매 step 새로 합성**한다 (`korean_split_dataset.py`
= 원본 repo `OnlineSplitFontSquare` 의 한글 서브클래스). 따라서 모든 샘플이 unique 이고
사실상 무한 데이터셋. 한 샘플 = `(style_img, style_text, gen_img, gen_text)` 4-tuple.

### (1) 폰트 = writer
- **한글 손글씨/디스플레이 폰트 83종** (`assets/fonts_korean_v2/train/` 의 .ttf).
  Eruku 의 "writer(필체)" 개념을 폰트 1종 = writer 1명으로 매핑.
  (초기 손글씨 폰트 + Google Fonts 한글(`tools_fetch_gf_korean.py` 수집)을 병합해 현재 83종)
- 렌더가 깨지는 폰트 3종(NMFClassic, GangwonEduSaeeum, KCCPakKyongni)은 제거됨.
- `fonts_charsets.json` 에 각 폰트의 charset 을 캐시 → 폰트가 못 그리는 글자(두부 □)를
  애초에 샘플하지 않아 tofu 방지.
- **held-out 검증 폰트 16종** (`assets/fonts_korean_unseen/`, `--val-fonts-dir` 기본값):
  학습에 안 쓴 폰트로 val loss 측정 → 스타일 일반화 확인.
- **영어 폰트 177종** (`assets/fonts_english/`, `--english-fonts-dir` 기본값):
  `--english-frac > 0` 일 때 라틴 폰트 데이터를 섞어 영어 스타일 다양성 보강 (`tools_fetch_gf_english.py` 로 수집).

### (2) 텍스트 = MixedLineSampler 로 합성한 가짜 라인
style 텍스트와 gen(target) 텍스트를 **각각 독립적으로**, 아래 비중으로 토큰을 섞어 1줄 생성
(`korean_split_dataset.py:65`, 원본 논문의 "English + random words" 를 한글로 확장):

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
- 데이터 sanity check: `uv run python korean_split_dataset.py --n 8 --out /tmp/ksplit`
  → 8개 style/gen pair + montage 저장.

### 원본 repo(`OnlineSplitFontSquare` + `TextSampler`) 대비 변경점

한글판은 원본 온라인 합성 파이프라인을 **서브클래스**(`KoreanSplitFontSquare`)로 재사용하되,
아래 4가지를 바꿨다. 근거는 각 항목의 코드 위치 참조.

| 구분 | 원본 repo | 한글판 (변경) | 이유 |
|---|---|---|---|
| **텍스트 소스** | `TextSampler`: 영어 단어만(nltk 6개 코퍼스 103,411단어), uni/bigram 빈도 가중 1종 sampler로 style·gen 동일 추첨 | `MixedLineSampler`: 한글0.45/영어0.20/숫자0.20/랜덤음절0.15 혼합 + 구두점·특수기호 부착. **폰트 charset 인지**(못 그리는 글자 미추첨 → tofu 방지) | 한글+숫자+기호 혼합 라인이 목표. 빈도편향 없이 전 음절 노출(랜덤음절) |
| **style/gen sampler** | 단일 sampler → style·gen 어절수 범위 동일 | style/gen **독립 sampler**(범위 분리 가능: Phase1 2~3/2~3, Phase2 1~8/1~32) | 조건(style)과 타겟(gen) 길이를 따로 통제 |
| **렌더 방식** | `[style\|gap\|gen]`을 **한 이미지로 합쳐 렌더** → 증강 → 고정 컬럼비로 분할 | style/gen을 **각각 따로 렌더**(경계 없음) | 원본은 회전·warp가 경계를 밀어 style↔gen 침범(잘림/잔상). 분리로 제거 (`korean_split_dataset.py:80-123`) |
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
- **`docs/DATA_GENERATION.md`** — 원본 `OnlineSplitFontSquare` 데이터 생성 전체 흐름(코드 단위).
- **`docs/PIPELINE_STEP_BY_STEP.md`** — 렌더→증강→split 을 단계별 이미지로 시각화.
  긴 텍스트에서 ①회전(`expand=False`)이 양끝을 잘라내고 ②warp 가 split cut 을 글자로 밀어넣는
  문제를 실측·정리(파이프라인은 **2~3어절 짧은 라인 가정** 위에서만 안전).
- **`docs/STYLE_LEN_BUG.md`** — ⚠️ **`get_model_inputs` 단위 버그**: `sl`(픽셀)을
  `style_img_embeds.shape[-1]`(latent=픽셀/8)과 그대로 `min` 비교해 **style/gen 이 실폭의 ~1/8
  (≈글자 1개)로 잘려** 조건으로 들어가던 문제. `* 8` 로 픽셀 환산해 수정(사용률 ~10% → ~100%,
  실측·before/after 이미지 포함). **이 수정 이전 체크포인트는 style 을 거의 못 봤으므로 재학습 권장.**
  - 재현: `PYTHONPATH=. uv run python docs/_verify_style_len.py` (사용률 ~100% 확인)

## 추론 / 시각화

```bash
# style ref 용 소규모 합성 세트 생성 (lines_json 포맷)
uv run python gen_korean_fontset.py --per-font-train 5 --per-font-val 0 --out data/ref_set

# 생성 결과 그리드: [스타일 ref | 정답 예시 | 생성 cfg=...]
CUDA_VISIBLE_DEVICES=0 uv run python infer_show.py \
  --ckpt finetune_runs/korean_p2/checkpoint_step_020000.pth \
  --lines-json data/ref_set/train_lines.json \
  --seed-text "오늘 날씨가 좋아서 친구와 공원에서 커피를 마셨다 2024년" \
  --cfgs 1.0 1.25 1.5 1.75 2.0 --n-writers 4 --out _debug/show.png

# style 텍스트(ref 전사) 제공 여부:
#   기본        → style ref 의 텍스트를 조건으로 함께 줌
#   --no-style-text → style 이미지만으로 스타일 파악(전사 미제공). 컬럼 라벨에 (no-stext) 표시.
#   둘을 비교하려면 같은 인자로 두 번 실행(--out 만 다르게):
uv run python infer_show.py --ckpt ... --out _debug/show_stext.png
uv run python infer_show.py --ckpt ... --out _debug/show_nostext.png --no-style-text
```

## repo 구조

```
# 학습
train_core.py              # 학습 공통 코어 (make_parser 전체 인자 + 학습 루프)
train_phase1.py            # Phase 1 진입점 (짧은 어절, batch128×2, 65000 step)
train_phase2.py            # Phase 2 진입점 (긴 줄, resume, +5000 step)
train_korean_from_en.py    # 영어 pretrained → 한글 직접 (Phase 1 생략)
train_eruku_continous.py   # (원본 repo 스타일 트레이너: wandb/accelerate/webdataset 기반, 참고용)
# 데이터
korean_split_dataset.py    # 온라인 한글 split 데이터셋 (+ CLI 샘플 덤프, --show-model-input)
gen_korean_fontset.py      # 오프라인 라인 생성기 + MixedLineSampler/build_pools (공용)
tools_fetch_gf.py          # Google Fonts 수집기 (--lang ko|en, 한/영 통합)
tools_font_audit.py        # 폰트 품질 감사: 미지원(두부)·malformed 글리프 스캔 (코퍼스 기준)
# 추론 / 평가  (infer_show.py 가 공유 라이브러리: load_model/gen_arr/render_in_font …)
infer_show.py              # 자기설명적 결과 뷰어 [스타일ref|정답|생성 cfg…] (--exclude-writers 로 폰트 제외)
infer_matrix.py            # 스타일×텍스트 매트릭스 뷰어 (1 style → 여러 gen_text)
infer_echo_compare.py      # echo(style==gen) 모델 나란히 비교 뷰어
eval_echo_metrics.py       # echo 정량 (MSE/SSIM/LPIPS/FID)
eval_cer.py                # echo CER (OCR 기반, 정렬무관·언어공정)
eval_all.py                # 통합 평가 배터리 (eval_cer+eval_echo_metrics 오케스트레이션)
# 모델 / 원본 코드
eruku_continuous_inf.py    # Emuru 모델 (forward/generate) — style-len 단위버그 수정됨
custom_datasets/           # 원본 repo 데이터 코드 (font_square 렌더/증강, tps)
models/                    # OrigamiNet 등
docs/                      # 파이프라인 상세/버그 분석 문서 + 재현 스크립트·이미지
assets/
  fonts_korean_v2/train/   # 한글 손글씨/디스플레이 폰트 83개 + fonts_charsets.json
  fonts_korean_unseen/     # held-out 검증 폰트 16개 (--val-fonts-dir 기본값)
  fonts_english/           # 영어(라틴) 폰트 177개 (--english-fonts-dir 기본값)
  fonts_label/             # NanumGothic (뷰어 라벨용)
  backgrounds/             # 종이 배경
  corpus/                  # korean_lines.txt(한 줄당 한 문장) / chars.txt / english_words.txt
```

## 지금까지의 실험 요약 (2026-06 기준)

| 실험 | 결과 |
|---|---|
| 공식 pretrained, 영어 생성 | 완벽 (문장 전체, in-style, EOG 정상) → 모델/추론 코드 정상 |
| lr 5e-5 fine-tune (online split) | **발산** — mse 진동, 전 토큰 blank 붕괴 |
| lr 1e-5 fine-tune 4K step | 안정 (loss 1.61→1.04). 영어 능력 보존 + 한글 폰트 style 적응 시작. 한글 glyph 는 아직 미생성 (숫자 조각부터 emerge) |

핵심 인사이트:
- **lr 1e-5 권장.** 5e-5 는 영어 init 에서 발산.
- 한글 glyph 생성은 byte→glyph 매핑의 **신규 학습**이라 scale 필요 (논문 Phase2 = 10M 샘플). 4K step(8K 샘플)은 0.1% 미만 — 장기 학습 필수.
- style ref 가 학습 분포(합성 폰트) 밖이면 생성이 붕괴(OOD). fine-tune 이 진행될수록 한글 폰트 prefix 에 적응.
- 모델은 SOG 전까지 style 텍스트 나머지를 이어 그린 뒤(target 앞 echo) gen 텍스트를 그림 — 뷰어는 입력 style 폭만큼 잘라 보여주므로 echo 가 생성부 앞에 보일 수 있음.
- 렌더가 깨지는 폰트 3종(NMFClassic, GangwonEduSaeeum, KCCPakKyongni)은 assets 에서 제거됨.

## 모델 성능 비교 (echo 정량, 2026-07)

측정 방식: **echo(style_text == gen_text)** — 텍스트 T 를 폰트로 깨끗이 렌더한 GT 를 style 로 주고 같은 T 를
생성, GT 와 비교. n=100 다양 길이(1~15어절), cfg=1.0, seen 폰트. 스크립트 `eval_echo_metrics.py`.
(↓ 낮을수록 좋음: MSE·LPIPS·FID / ↑ 높을수록: SSIM. **굵게 = 열 최고**)

### 한글 (echo, n=100)

| 모델 (step) | MSE↓ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| 구 transform 20k | 0.195 | 0.364 | 0.459 | 75.6 |
| **v2 20k** | **0.187** | **0.377** | **0.450** | **70.7** |
| v2 24k | 0.207 | 0.344 | 0.464 | 81.7 |
| v2 27k | 0.194 | 0.370 | 0.462 | 76.6 |
| v2 30k | 0.208 | 0.336 | 0.450 | 75.0 |

### 영어 (echo, n=100, pretrained 포함)

| 모델 | MSE↓ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| pretrained (영어 원조) | **0.145** | **0.503** | 0.421 | 84.8 |
| 구 transform 20k | 0.168 | 0.455 | 0.367 | 54.4 |
| **v2 20k** | 0.162 | 0.468 | 0.324 | **52.5** |
| v2 30k | 0.170 | 0.453 | **0.318** | 57.3 |

### 정성 관찰 (montage/CER)

- **완성도(문장 끝맺음)**: pretrained 는 긴 표준영어 완전완성(16~17어절)에 강함. v2 는 긴 **한글**을 30k 에서
  4/4 완성(20k 은 반복붕괴)했으나 긴 **영어**는 아직 반복붕괴 → 학습분포상 긴영어 부족(전체의 ~10%).
- **출력 스타일**: v2 는 gen 을 **흰배경·곧은글자(무휨)** 로 생성(새 데이터 정책). 구 모델은 휨/배경 잔존.
- **트레이드오프**: 한글·폰트 스타일적응(v2) ↔ 긴 영어 완성(pretrained). v2 20k 가 파인튜닝 중 fidelity 정점,
  30k 는 긴한글 완성↑ 이나 fidelity 소폭↓(과적합).
- **주의(지표 해석)**: 픽셀(MSE/SSIM)은 글자 위치/간격 정렬오차에 민감해 절대값 과소평가·언어간 비교 부적절;
  FID(n=100)는 노이즈 큼. **OCR 기반 CER**(`eval_cer.py`, 정렬무관·언어공정·완성도 반영)로 정밀 재측정 진행 중.
- 도구: `infer_show.py`(그리드), `infer_matrix.py`(스타일×텍스트 매트릭스), `infer_echo_compare.py`(모델 나란히),
  `eval_echo_metrics.py`/`eval_cer.py`(정량).

## 요구 사양

- CUDA GPU ≥ 24 GB (batch 2, max-img-len 2048 기준; T5-large 705M trainable)
- 디스크: repo ~250 MB + pretrained 2.9 GB(자동 다운로드) + 학습 ckpt 개당 ~8 GB
- TPS C++ 백엔드는 선택 (`custom_datasets/font_square/tps/build.sh`, `uv sync --extra tps` 후) — 없으면 NumPy fallback 으로 동작
