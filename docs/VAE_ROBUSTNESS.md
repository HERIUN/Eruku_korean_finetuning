# VAE 강건성 & 실제 손글씨 미러링 (2026-07-23)

full_mse VAE + 단문강조 재적응으로 **폰트-렌더 입력**의 한글 생성 CER 은 크게 내렸다
(→ README 「Eruku 재적응 결과」). 하지만 **실제 스캔 손글씨 라인**을 style 로 주면 일부 모델이
생성을 못 하고 blank/runaway 로 무너졌다. 그 원인분석과 두 가지 별개 결함을 여기 정리한다.

두 결함은 **서로 다른 축**이다 — 하나는 디코더(복원), 하나는 인코더(조건):

| | 결함 | 축 | 관측 | 상태 |
|---|---|---|---|---|
| ① | **프로토타입 붕괴** (prototype collapse) | decoder / 복원 | 입력 잉크농도 무시하고 항상 crisp 순검정으로 snap | 학습설계 결함 (수정=재학습 필요, 미적용) |
| ② | **실제 손글씨 runaway** | encoder / 조건 | style 이미지 여백 → latent std OOD → T5 runaway/blank | **추론 bbox 크롭으로 해결(재학습 불필요)** |

---

## ① 프로토타입 붕괴 — VAE 는 복원이 목적이지 과선명이 목적이 아니다

우리 decoder/full_mse VAE 는 입력 잉크농도를 **완전히 무시하고 항상 crisp 순검정으로 snap** 한다.

**결정적 실험** (`scratchpad/vae_sharpen.py`): 입력 ink floor 를 0.0/0.2/0.4/0.6 으로 올리며 출력 최소 ink 관측 —

| 입력 ink floor | 원본 emuru 출력 | 우리 VAE 출력 |
|---|---|---|
| 0.60 | 0.517 (**입력 따라감 = 충실**) | **0.000 (고정)** |

- **원인**: 학습 타깃 `bw` 가 항상 crisp binary(`base64*2-1`) + pure-L1 + 폰트 99종 → VAE 가
  농도/대비 차원을 latent 에서 버림. **버그가 아니라 학습설계 결함.**
- **함의**:
  1. headline clean-MSE −83% 의 상당 부분이 진짜 fidelity 가 아니라 "crisp 타깃에 snap" 이득.
  2. 실제 손글씨(희미·얇음) 복원은 오히려 **+89% 악화**(overshoot — 얇은 획을 뭉갠 검정으로).
  3. Eruku 가 필압·농담 style 조건을 못 받음(appearance 소실) → real-HW conditioning OOD 에 기여.
- **수정 방향(재학습 필요, 미적용)**: 타깃에 **입력과 매칭된 잉크 alpha** 부여(농담 보존 denoising —
  배경/블러만 제거하고 잉크농도는 유지). `custom_datasets.korean_aux._make_rgb`/`__getitem__` 에서 alpha 를
  한 번 뽑아 **rgb 입력과 bw 타깃이 공유**하도록. 단 VAE 가 바뀌면 latent 이동 → Eruku 재적응 필요.
- **대안**: 원본 emuru VAE 는 이미 농담보존 + real-HW 복원이 우수 → 재설계 대신 **원본 유지** 또는
  매우 약한 finetune 도 선택지.
- 진단도구: `experiments/vae_hw_recon.py`(실제 HW 복원 MSE/SSIM), `scratchpad/vae_sharpen.py`(농도 snap 테스트).

---

## ② 실제 손글씨 runaway — 근본원인은 style 이미지 여백, 수정은 ink bbox 크롭

실제 스캔 라인을 인코딩하면 latent std 가 **폰트(0.014~0.022)의 6~29배(0.13~0.42)** 로 튀어
T5 조건이 OOD → runaway(EOG 못 뱉음)/blank.

- 원인은 appearance OOD 도, ①의 농담 붕괴도 아니고 **이미지 여백(주로 좌우 마진)** 이다.
  `_img_encode` 가 `.latent_dist.sample()`(stochastic)이라 seed 의존도 얹히지만, `.mode()` 로 바꿔도
  안 고쳐진다(mode 도 OOD).
- **수정: style 을 ink bbox 로 타이트 크롭 후 인코딩** → latent std 0.4 → **0.02(폰트 수준)** → 생성 안정.
- ★**단순 min/max bbox 는 실패** — 가장자리 노이즈·스캔 테두리에 취약(예 line_008: 좌우 검은 세로
  테두리 + 상하 stray 3px → bbox 가 이미지 전체 → 크롭 무효 → blank 유지). **밀도 기반 bbox 필수**
  (행/열 잉크비율 임계 범위만 취해 sparse 노이즈 무시 + 가장자리 solid 테두리 strip).
- **임계 비대칭**(중요): 행(세로)은 낮게(`dens_row≈0.006`) — 글자 위아래 성긴 획(받침·삐침)까지
  살려 안 잘리게. 열(가로)은 `dens_col≈0.03` — 좌우 여백 제거. 테두리 감지는 회색 anti-alias 까지
  (`bthr<215`), 텍스트 밀도는 진하게(`thr<180`).
- **검증**: mirror montage 에서 밀도 bbox 적용 시 4모델 × 8샘플 **32/32 전부 legible**
  (단순 bbox 는 line_008 만 실패 → 밀도 bbox 로 6.78% 생성 회복).
- 구현: `experiments/mirror_compare.py`(`bbox_crop_gray` 밀도기반, `--bbox-crop` 기본 on).
  진단 scratchpad: `diag_bbox.py` / `diag_robust_bbox.py` / `diag_l008.py`.
- ⚠️ **미결**: 정식 추론경로(`infer.show.load_style` 및 eval 스크립트들)에 이 bbox 크롭을 baked-in 할지
  아직 미적용 — 현재는 mirror montage 에만 들어있다.

### 왜 bbox 크롭이 통하나 — 학습 갭

`custom_datasets.korean_split` 학습 파이프라인은 style 을 `TailorTensor(ink 타이트크롭) → ImgResize(64)` 로
**항상 텍스트가 프레임을 꽉 채운 상태**로만 만든다 → 스케일/여백 변화가 전무. 그래서 모델은
"작은 글자 + 큰 여백" style 을 **한 번도 못 봤고**, 실제 스캔(여백 있음)이 OOD → runaway 로 이어진다.
bbox 크롭이 통하는 근본 이유 = 추론 입력을 이 학습분포(텍스트가 프레임 채움)에 맞추는 것.

대응 2안:
1. **추론 bbox 크롭**(즉효, 현재) — 재학습 불필요.
2. **학습 시 style 스케일/여백 augmentation**(native 강건) — 재학습 필요, 미구현.
   (한때 style scan-aug 를 시도했으나 되돌림. 근본 레버는 ①의 VAE prototype collapse 쪽.)

### runaway 는 "unseen 스타일" 이 아니라 "실제 스캔 특성(저대비·얇음·노이즈)" 특정

held-out AI손글씨 폰트 4종(`assets/custom_hw_fonts`, **평가전용**) 생성은 runaway 없이 전부 legible
(`experiments/gen_heldout.py`, montage `finetune_runs/eruku_short_fullmse/_eval/gen_heldout_fonts.png`).
폰트-렌더(고대비)는 held-out 이어도 잘 된다 → 문제는 **스타일 신규성이 아니라 입력분포(스캔 vs 렌더)** 다.

---

## 결과 이미지

- 실제 손글씨 20문장 × 4모델 미러링: `finetune_runs/eruku_short_fullmse/_eval/mirror_all20.png`
  (열 = 실제 손글씨(목표=style) | 기존 원본VAE | decoder VAE 교체 | full_mse s15k | 현재 최적 s30k)
- 재적응 전후(s15000 vs s30000) 생성 비교: `finetune_runs/eruku_short_fullmse/_eval/gencmp_*.png`
- held-out 폰트 생성: `finetune_runs/eruku_short_fullmse/_eval/gen_heldout_fonts.png`

## 실행 환경

이 repo 는 **`.venv/bin/python`** 으로 실행한다(torch 2.9+cu128, diffusers/skimage/transformers 포함).
시스템 `python`(miniconda base)에는 numpy 조차 없다. GPU 는 2번만 여유(`CUDA_VISIBLE_DEVICES=2`).
