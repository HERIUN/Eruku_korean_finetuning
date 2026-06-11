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

# 3-A. 한글 Phase 1 — 한글 glyph 본 학습 (짧은 2~4 어절 pair)
#      virtual 256 = batch 16 × accum 16, lr 1e-4, wd 1e-2 (논문 레시피)
CUDA_VISIBLE_DEVICES=0 uv run python train_korean.py \
  --online-split --style-words 2 4 --gen-words 2 4 \
  --text-dropout 0.05 \
  --batch-size 16 --grad-accum 16 --lr 1e-4 \
  --max-steps 20000 --save-every 1000 --log-every 10 \
  --save-samples 16 \
  --out finetune_runs/korean_p1

# 3-B. 한글 Phase 2 — 긴 줄 일반화 (Phase 1 ckpt 에서 resume)
#      논문 그대로: batch 2 × accum 128 = virtual 256, 5000 iter
CUDA_VISIBLE_DEVICES=0 uv run python train_korean.py \
  --online-split --style-words 1 8 --gen-words 1 32 \
  --text-dropout 0.05 --style-text-dropout 0.10 \
  --batch-size 2 --grad-accum 128 --lr 1e-4 \
  --max-steps 5000 --save-every 250 --log-every 10 \
  --resume finetune_runs/korean_p1/checkpoint_last.pth \
  --out finetune_runs/korean_p2
```

왜 두 단계인가: 논문에서 **Phase 1 (65000 iter × 256 = 16.6M 샘플, 짧은 단어쌍)** 이
byte→glyph 매핑을 만들고, **Phase 2 (5000 iter = 1.28M, 긴 줄)** 는 길이 일반화만 담당.
한글 glyph 는 영어 pretrained 에 없는 **신규 능력**이라 Phase-1 형태(짧은 샘플 = 같은
시간에 더 많은 glyph 감독)로 먼저 가르치는 것이 효율적. 단 from-scratch 는 불필요 —
VAE-latent 디코딩·자기회귀·style 메커니즘은 영어 pretrained 에서 전이됨.

- `--max-steps` / `--save-every` / `--log-every` 는 **virtual step**(optimizer 업데이트) 기준
- 노출량: Phase 1 20K step = 5.1M 샘플 (논문 65K 의 1/3 — 영어 전이 가정, 생성 보며 연장),
  Phase 2 5000 step = 1.28M (논문과 동일). 논문은 10M 고정 데이터셋에서 1.28M 만 썼지만
  우리는 온라인 무한 생성이라 모든 샘플 unique (resume 시에도 seed offset 으로 중복 방지)
- 24GB 기준 측정 속도: Phase 1 (batch16) virtual step ~10s → 20K ≈ 55h,
  Phase 2 (batch2) virtual step ~44s → 5K ≈ 60h
- 빠른 sanity check (accumulation 없이): `--grad-accum 1 --lr 1e-5 --max-steps 4000`
  — **주의: virtual batch 가 작으면 lr 도 낮춰야 함** (batch 2 + lr 5e-5 발산 확인됨,
  논문 lr 1e-4 는 virtual 256 기준)

첫 실행 시 HuggingFace 에서 `google-t5/t5-large`(config), `google/byt5-small`(tokenizer),
`blowing-up-groundhogs/emuru_vae`, 그리고 **영어 pretrained weight**
(`blowing-up-groundhogs/eruku` 의 `pytorch_model.bin`, ~2.9GB) 를 자동 다운로드하므로
인터넷이 필요합니다. OrigamiNet OCR ckpt 는 불필요 (alpha=1.0 → OCR loss 미사용,
공식 HF 릴리즈에도 OCR 모듈 없음).

### 학습 산출물 (`--out` 디렉토리)

- `train_config.yml` — 실행마다 전체 인자+타임스탬프가 **run 히스토리**로 누적 기록
- `train_loss.csv` — `step,loss,mse,ce,it_s,timestamp` (log-every 마다, resume 시 이어 씀)
- `checkpoint_step_XXXXXX.pth` — model + optimizer + step (resume 용)
- `train_samples/` — `--save-samples N` 지정 시 학습 데이터 샘플 + montage

### 이어 학습 (resume)

```bash
uv run python train_korean.py ... \
  --resume finetune_runs/korean_p2/checkpoint_step_004000.pth --max-steps 200000
```

resume 시 이전 run 의 `train_config.yml` 과 인자가 다르면 `[config WARN]` 으로 알려줍니다
(lr/batch 등 바뀐 채 이어 학습하는 실수 방지).

## 데이터 파이프라인 (온라인, 디스크 0)

`korean_split_dataset.KoreanSplitFontSquare` = 원본 repo 의 `OnlineSplitFontSquare` 서브클래스.

1. style/gen 텍스트를 **각각 다른 어절수 범위**로 샘플 (`MixedLineSampler`:
   한글 0.45 / 영어 0.20 / 숫자 0.20 / **랜덤음절 0.15** + 구두점·특수기호).
   랜덤음절 = KS X 1001 2,350자(전 폰트 교집합)에서 균등 추첨한 1~4글자 가짜 단어
   (논문의 "random words" 대응) — 코퍼스 빈도 편향 없이 전 음절을 단어 맥락으로 노출
   (2,000라인 샘플링 시 2,347/2,350 음절 등장 확인)
2. 같은 폰트로 두 텍스트를 렌더 → concat → 증강(rotation, TPS warp, blur, 종이배경, ink jitter, dilation 등) → 폭 기준 split → `style_img` / `gen_img`
3. 폰트 = writer 1명. `assets/fonts_korean_v2/train/` 61개 손글씨 폰트
4. `fonts_charsets.json` 의 전 폰트 charset union 으로 tofu 방지

논문 레시피 매핑:
- `--text-dropout 0.05` = p_uncond (style+gen 텍스트 모두 drop → CFG 학습)
- `--style-text-dropout 0.10` = p_drop (Phase 2 전용, style 텍스트만 drop)
- Phase 1: `--style-words 2 4 --gen-words 2 4` / Phase 2: `--style-words 1 8 --gen-words 1 32`
- virtual batch 256: `--grad-accum` × `--batch-size` = 256 (논문 lr 1e-4 는 이 기준)
- 논문 iteration: Phase 1 = 65000 (16.6M 샘플), Phase 2 = 5000 (1.28M 샘플)
- target 패딩 = visual `<EOG>`: 이미 구현되어 있음 — specials 패딩값 1(=EOG) +
  forward 에서 EOG embedding 치환 + CE 로 "gen 종료 후엔 EOG 만" 학습

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
```

## repo 구조

```
train_korean.py            # 학습 런처 (online-split / resume / dropout 옵션)
korean_split_dataset.py    # 온라인 한글 split 데이터셋 (+ CLI 샘플 덤프)
gen_korean_fontset.py      # 오프라인 라인 생성기 + MixedLineSampler/build_pools (공용)
infer_show.py              # 자기설명적 결과 뷰어
eruku_continuous_inf.py    # Emuru 모델 (forward/generate)
custom_datasets/           # 원본 repo 데이터 코드 (font_square 렌더/증강, tps)
models/                    # OrigamiNet 등
assets/
  fonts_korean_v2/train/   # 손글씨 폰트 61개 + fonts_charsets.json
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

## 요구 사양

- CUDA GPU ≥ 24 GB (batch 2, max-img-len 2048 기준; T5-large 705M trainable)
- 디스크: repo ~250 MB + pretrained 2.9 GB(자동 다운로드) + 학습 ckpt 개당 ~8 GB
- TPS C++ 백엔드는 선택 (`custom_datasets/font_square/tps/build.sh`, `uv sync --extra tps` 후) — 없으면 NumPy fallback 으로 동작
