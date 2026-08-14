# 실험 로그 — 측정 결과 전문

README 에서 분리한 **측정 기록**이다. 코드에 반영된 결론만 보려면
[README 「결과 요약」](../README.md#결과-요약) 을 보면 된다. 여기엔 그 결론을 뒷받침하는
수치·표·재현 커맨드를 시간순으로 남긴다.

| 절 | 주제 | 한 줄 결론 |
|---|---|---|
| §1 | 초기 실험 (2026-06) | lr 5e-5 는 영어 init 에서 발산, 1e-5 안정. 한글 glyph 는 신규 학습이라 scale 필요 |
| §2 | echo 정량 비교 (2026-07) | v2 20k 가 fidelity 정점. 픽셀 지표는 정렬오차에 민감 → CER 로 재측정 |
| §3 | VAE 한글 수용성 (2026-07) | 한글 재구성 오차 = 영어의 ~12배. **이것이 한글 fidelity 하드상한** |
| §4 | VAE 한글 적응 (2026-07) | full(enc+dec) + HTR off(pure-L1) 가 최선, roundtrip MSE −83% |
| §5 | Eruku 재적응 (2026-07) | 단문강조 재적응으로 한글 CER 0.211 → 0.127, 단문 0.54 → 0.29 |
| §6 | VAE 강건성 | 실제 손글씨는 여백이 latent std 를 6~29배로 튀김 → ink bbox 크롭 |
| §7 | 종성 획 손실 (2026-08) | 겹받침 음절정확도가 홑받침보다 22%p 낮음 (진행 중) |

---

## 1. 초기 실험 요약 (2026-06)

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

## 2. 모델 성능 비교 — echo 정량 (2026-07)

측정 방식: **echo(style_text == gen_text)** — 텍스트 T 를 폰트로 깨끗이 렌더한 GT 를 style 로 주고 같은 T 를
생성, GT 와 비교. n=100 다양 길이(1~15어절), cfg=1.0, seen 폰트. 스크립트 `eval/echo_metrics.py`.
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
  FID(n=100)는 노이즈 큼. **OCR 기반 CER**(`eval/cer.py`, 정렬무관·언어공정·완성도 반영)로 정밀 재측정 진행 중.
- 도구: `infer/show.py`(그리드), `infer/matrix.py`(스타일×텍스트 매트릭스), `infer/echo_compare.py`(모델 나란히),
  `eval/echo_metrics.py`/`eval/cer.py`(정량).

## 3. 한글 학습의 하드 상한 — VAE 한글 수용성 (2026-07)

en2ko fine-tune(영어 pretrained→한글)에서 한글 CER 이 step **~11k 부근에서 plateau** 하고, 그 이상 학습하면
val loss 만 오르고(과적합) 영어가 소폭 퇴보한다. 아무리 더 돌려도 한글 fidelity 가 어느 선 위로 안 올라간다.
원인은 학습 부족이 아니라 **동결 VAE 의 한글 재구성 한계**다.

모델은 픽셀이 아니라 **VAE latent** 만 조건·타깃으로 쓴다(latent 1열 = 8차원). 따라서 `decode(encode(x))` 가
못 담는 디테일은 모델이 애초에 조건으로 볼 수도, 타깃으로 배울 수도 없다 = **한글 학습의 하드 상한**.

같은 폰트(NanumGothic)로 한글/영어를 VAE 왕복(encode→decode)시킨 재구성 오차 — 근거 스크립트 `docs/_vae_capacity.py`:

| | 평균 MSE↓ | 평균 SSIM↑ | 최악 사례 |
|---|---|---|---|
| **한글** | **0.0165** | **0.906** | "형형색색 꽃들이 활짝" → SSIM **0.828** (밀집 음절 뭉개짐) |
| **영어** | 0.0014 | 0.984 | 3개 모두 SSIM ~0.98 (거의 완벽 복원) |

한글 재구성 오차가 영어의 **약 12배**. 특히 획이 밀집한 음절(형·꽃·활짝)에서 심하게 흐려진다 —
초·중·종성이 한 칸에 2D로 쌓이는 한글이, latent 폭당 8차원 표현에서 영어(선형 단순 글자)보다 불리하기 때문.

![VAE 한글 vs 영어 재구성](vae_korean_vs_english.png)

*각 예: 원본 vs VAE복원. 영어는 원본과 거의 동일, 한글은 밀집 음절일수록 뭉개짐(=모델이 볼 수 있는 정보의 상한).*

**함의 — 한글 fidelity 를 더 올리려면 학습이 아니라 VAE 를 손대야 한다:**
- (a) **VAE decoder 파인튜닝** — encoder/latent 고정(T5 인터페이스 유지) → 출력 렌더 충실도만 개선. **저위험**.
- (b) **입력 해상도↑** (h=64→96, latent 세로 8→12로 용량↑). 단 `query_emb`/`t5_to_vae` 재init 필요.
- (c) **한글로 VAE encoder+T5 공동 재학습** — conditioning 정보 상한 자체를 올림. **고위험**(pretrained 강점 재학습).

## 4. VAE 한글 적응 실험 (2026-07) — **full(enc+dec) + HTR off 가 최선**

위 (a)/(c) 를 실제로 학습해 비교. 파이프라인: 한글 HTR finetune(`train/aux_htr.py`) → VAE finetune(`train/vae_korean.py`) →
교체 검증(`infer/show.py`/`eval/cer.py --vae-checkpoint`). clean-bw roundtrip MSE(고정 probe, `experiments/vae_recon.py`):

| VAE | 일반 한글 | 극단 폰트 | 영어 |
|---|---|---|---|
| 원본 emuru | 0.0188 | 0.0197 | 0.0020 |
| decoder-only + HTR | 0.0120 (−36%) | — | 0.0032 |
| **full + HTR(0.3)** | 0.0089 (−53%) | 0.0183 | 0.0037 ⚠️영어 **악화** |
| **full + HTR off (pure-L1)** ⭐ | **0.0031 (−83%)** | **0.0128** | **0.0012** |

**핵심 발견 — VAE 한글 적응은 `--train-part full --htr-weight 0`(HTR 끔)이 결과가 제일 좋다:**
- HTR(OCR 가독성) loss 는 **MSE 를 최적화하지 않는다** — "정확한 글자로 읽히게"(프로토타입) 밀어서 그 폰트의 정확한 획에서 멀어짐 → 픽셀 MSE↑. 텍스트는 흑백 고대비라 **L1 만으로 이미 획이 또렷**해서 HTR 가독성 기여가 중복이고, 프로토타입 편향·1채널 용량 경쟁 비용만 남는다.
- 특히 **한글 HTR 은 영어를 악화**시킴(공유 decoder 를 한글 획으로 편향, 0.0020→0.0037). pure-L1 은 스크립트 중립이라 영어도 개선.
- pure-L1 이라도 **흐릿해지지 않음** — 극단 폰트(초굵은/붓글씨)에서도 획을 선명히 살림(고대비 → L1 이 엣지 보존).
- 원본 Emuru 가 HTR 로 이득 본 건 배경노이즈 **디노이징** 태스크 + 영어HTR·영어데이터 정합 + 10만폰트 맥락이라, 깨끗한 한글 폰트 복원인 우리와 상황이 다름.

권장 레시피(VAE finetune): `--train-part full --htr-weight 0 --kl-weight 1e-6`, warm-start(decoder-only→full 순), 지표는 held-out roundtrip MSE.
**주의**: `train-part full` 은 latent 공간이 바뀌므로 Eruku 를 **기존 한글 ckpt 에서 resume + `train/core.py --reset-vae`**(ckpt 의 옛 vae.\* strip) 로 재적응해야 한다. decoder-only 는 latent 불변이라 재학습 없이 `--vae-checkpoint` 교체만으로 사용 가능.
한계: held-out 극단 폰트는 여전히 높음(폰트 다양성 부족, 원본 10만 vs 우리 99). 증강·균일빈도로는 오히려 악화 → 폰트 수집이 유일한 진짜 레버.

### log_var(학습형 관측 노이즈 가중) — 논문 아닌 원본 *코드* 관례, 우리 세팅선 무익 → [`docs/VAE_LOGVAR.md`](VAE_LOGVAR.md)

원본 `models/autoencoder_loss.py` 의 `nll/exp(log_var)+log_var` 항(LDM/CompVis Kendall식 학습형 recon 가중, **논문엔 없음**)을 `--logvar-weighting` 로 복원해 통제실험. 원본 emuru VAE 에서 동일 seed·데이터·예산(full·pure-L1·15000step)으로 **log_var 유무만** 다르게(`run_logvar_clean.sh`):

| VAE (s15000) | 일반 한글 | 극단 폰트 | 영어 |
|---|---|---|---|
| 원본 emuru | 0.0188 | 0.0197 | 0.0020 |
| control (logvar off) | **0.0038** | **0.0131** | **0.0014** |
| logvar on | 0.0037 (동률) | 0.0138 (+5%) | 0.0015 (+7%) |

**결론 — log_var 무익, 기본 off**: 그 수학적 실익(관측 노이즈 MLE 자기보정·grad 스케일 정규화·다중 loss 항 밸런싱)은 모두 "여러 항을 섞고 KL 이 유의미할 때" 발동하는데, 우리는 **KL=1e-6(무시수준)·단일 L1 항·Adam(이미 grad 정규화)** 이라 발동 여지가 없다. 최종 `lv=-1.481` 로 이론 종착점 `s*=log L≈-3.5` 에 15k step 내 수렴도 못 함(inert). ko 동률·hard/en 미세 열화로 실측 확인. → `--logvar-weighting` 플래그만 재현용 보존, 기본 off.

### loss 항 ablation (pyramid / HTR / 예산) — 결론: **pure-L1 이 최선, 예산이 진짜 레버**

`--pyramid-weight`(Laplacian band-pass = 고주파 대역 가중, 획 엣지를 더 세게 감독하려는 시도) 와
HTR 가중치를 조합해 **같은 seed·데이터**로 통제 비교. 지표는 학습 중 held-out **폰트** val
roundtrip(각 런 `vae_val.csv` 마지막 행, `--val-font-frac 0.1`, seed 42):

| 런 (step) | train-part | loss 항 | 한글 MSE↓ / SSIM↑ | 영어 MSE↓ |
|---|---|---|---|---|
| `vae_orig_dec_pureL1` (15k) | dec | L1 | 0.00972 / 0.9514 | 0.00205 |
| `vae_orig_dec_pyr` (15k) | dec | L1+pyr | 0.00997 / 0.9382 | 0.00191 |
| `vae_orig_pureL1` (15k) | full | L1 | 0.00713 / 0.9597 | 0.00196 |
| `vae_orig_pureL1_logvar` (15k) | full | L1+logvar | 0.00828 / 0.9539 | 0.00189 |
| `vae_orig_full_L1_htr01` (15k) | full | L1+htr 0.1 | 0.01048 / 0.9364 | 0.00266 |
| `vae_orig_full_pyr_htr01` (15k) | full | L1+pyr+htr 0.1 | 0.00894 / 0.9438 | 0.00198 |
| `vae_orig_full_pyr` (30k) | full | L1+pyr | 0.00645 / 0.9591 | **0.00143** |
| `vae_orig_full_pyr` (45k, 30k에서 resume 연장) | full | L1+pyr | 0.00711 / 0.9558 | 0.00140 |
| **`vae_korean_full_mse` (28k)** ⭐ 릴리즈 | full | L1 (디렉토리명만 mse) | **0.00656 / 0.9647** | 0.00161 |
| `vae_orig_full_htr03_30k` (30k, 완료) | full | L1+htr 0.3 | 0.01156 / 0.9263 | 0.00323 |

- **pyramid 는 이득 없음** — 같은 15k 예산에서 pure-L1 과 사실상 동률이거나 약간 나쁨
  (dec 0.00997 vs 0.00972, SSIM 은 −1.3%p). 텍스트는 이미 흑백 고대비라 L1 자체가 엣지를
  보존하므로 고주파 가중이 **중복**이다. 유일한 쓸모: htr 을 켠 조합에서 그 해악을 일부 상쇄
  (0.01048 → 0.00894) — 즉 "htr 로 흐려진 걸 pyr 로 되돌리는" 구조라, htr 을 끄면 필요 없다.
- **htr 은 예산을 30k 로 늘려도 열위**(완료 s30000 에서 0.01156, 영어도 0.0032) → 위 「HTR off」 결론 재확인.
- **예산은 듣지만 30k 부근에서 포화**: 동일 pure-L1 계열에서 15k 0.00713 → 28~30k 0.0065 로 내려간 뒤,
  그 이상은 더 안 내려간다.
- ⚠️ **resume 로 예산을 늘리면 오히려 퇴보할 수 있다.** `full_pyr` 을 s30000 에서 45k 까지 이어 돌린 결과
  0.00645 → s36000 0.00736 → s45000 0.00711 로, 15k step 을 더 쓰고도 **자기 최적점을 회복하지 못했다**.
  resume 는 Adam 상태·lr 스케줄을 새로 시작하므로 좋은 지점에서 밀려난다. 연장할 거면 처음부터 긴 예산으로
  한 번에 돌리고, 중간 최적 ckpt 는 반드시 보존할 것.
- 30k 예산에서 `full_pyr` 과 `full_mse` 는 한글 동급(0.00645 vs 0.00656)이고 SSIM 은 full_mse 가
  낫다 → **릴리즈는 `full_mse` s28000 유지**(교체 시 Eruku 재적응이 다시 필요한데 이득이 없음).

## 5. Eruku 재적응 (2026-07) — full_mse VAE + 단문강조로 단문 병목 해소

full_mse VAE 는 재구성 상한을 크게 올렸지만(roundtrip MSE −83%), 그 상한이 **실제 생성으로 발현되려면 T5 를
새 latent 공간에 재적응**시켜야 한다(full 은 latent 이동 → Eruku resume 필수). 진단 결과 생성 병목은 VAE decode 가
아니라 **T5 의 단문 latent 예측**이었다(한글 단문 CER ~0.5, 장문은 ~0.05). 이를 겨냥해 단문 비중을 높인 재적응을 수행:

```
# eruku_readapt_fullvae(full_mse VAE 내장) s15000 에서 resume, 단문강조
CUDA_VISIBLE_DEVICES=2 .venv/bin/python train/korean_from_en.py \
  --resume finetune_runs/eruku_readapt_fullvae/checkpoint_step_015000.pth \
  --out finetune_runs/eruku_short_fullmse \
  --gen-words 1 12 --style-words 1 8 --max-img-len 4096 \
  --extra-steps 15000 --batch-size 4 --grad-accum 8 --lr 5e-5 \
  --english-frac 0.15 --num-workers 16 --save-every 2000 --val-every 2000
```

**평가 = `eval/htr_cer.py`** (한글 HTR 리더, 렌더바닥 CER ~0.00 — easyocr 은 한글 GT 조차 0.236 로 saturated 해 못 씀).
n=100, cfg=1.0, coherent:

| 버킷 | decoder-only | full_mse (재적응 전) | **full_mse + 단문재적응 s30000** |
|---|---|---|---|
| 한글 **전체** | 0.203 | 0.211 | **0.127 (−40%)** |
| 한글 **단문(1~3)** | 0.497 | 0.540 | **0.286 (거의 절반)** |
| 한글 중문(4~8) | — | — | 0.049 |
| 한글 장문(9~15) | — | — | 0.059 |
| 영어 전체 | 0.084 | 0.056 | **0.045 (회귀 없음)** |

**핵심 결론**:
- 생성 품질 = **VAE(상한) × T5(상한 근접도)**. full_mse VAE 로 상한을 올린 뒤 단문강조 재적응이 T5 병목을 잡아, 단문 한글 CER 을 **0.54→0.29 로 절반** 냈다. 영어도 회귀 없이 개선.
- **best ckpt = `finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth`**. val_mse 는 s20000(0.100)이 s30000(0.113)보다 낮지만, 실제 생성 CER 은 s30000 이 우수(전체 0.127 vs 0.138, 단문 0.286 vs 0.317) — **val_mse 노이즈는 생성품질과 무관**하므로 CER 로 고를 것.
- 복잡획(밝·붉·닭·삶·값·짧…) 단문 생성 전후 비교: `finetune_runs/eruku_short_fullmse/_eval/gencmp_{ko_complex_short,ko_short,ko_long}.png` (재적응 전 s15000 vs 후 s30000, 동일 VAE — 차이는 순수 T5 개선분). 스크립트 `experiments/gen_compare.py`.

## 6. VAE 강건성 & 실제 손글씨 미러링 → [`VAE_ROBUSTNESS.md`](VAE_ROBUSTNESS.md)

폰트-렌더 입력은 잘 되지만 **실제 스캔 손글씨 style** 에서 드러난 두 별개 결함을 정리한 문서:
- **① 프로토타입 붕괴**(decoder/복원): VAE 가 입력 잉크농도를 무시하고 항상 crisp 순검정으로 snap →
  headline MSE 이득의 일부가 "crisp 타깃 snap" 이고 실제 손글씨 복원은 악화. 수정=농담보존 타깃(재학습 필요).
- **② 실제 손글씨 runaway**(encoder/조건): style 이미지 **여백** 이 latent std 를 폰트의 6~29배로 튀겨 T5 OOD →
  runaway/blank. **수정=밀도 기반 ink bbox 크롭**(재학습 불필요, `experiments/mirror_compare.py`). 근본은 학습이
  style 을 항상 프레임 꽉 채운 상태로만 본 갭.

## 7. 종성(받침) 획 손실 — 특히 **겹받침** (2026-08, 진행 중)

받침 있는 음절에서 획이 뭉개지는 현상을 정량화했다. 상세·중간 실험은
[`docs/JONGSEONG_STROKE_LOSS.md`](JONGSEONG_STROKE_LOSS.md), 측정 스크립트는
`experiments/jong_stroke_loss.py`(VAE 픽셀 정밀) / `jong_gen_loss.py`(생성, 자모 단위).

**① VAE 재구성** (roundtrip 은 입력과 픽셀 정렬 → 음절 창을 잘라 정확히 측정. 폰트 16종 × 120라인,
n=23,039 음절. `missing` = GT 잉크 중 복원에서 사라진 비율):

| 종성 유형 | n | missing↓ | IoU↑ | 받침자리(아래 40%) missing↓ |
|---|---|---|---|---|
| 없음 | 3,312 | 0.054 | 0.899 | 0.048 |
| 홑받침 | 17,279 | 0.077 | 0.863 | 0.081 |
| 쌍받침 ㄲㅆ | 1,280 | 0.098 | 0.835 | **0.131** |
| 겹받침 ㄳㄵㄺ… | 1,168 | 0.100 | 0.828 | 0.115 |

종성 없음 대비 홑 +43% / 쌍 +81% / 겹 +85%. 받침 자리로 좁히면 격차가 **2.7배**로 벌어져,
손실이 음절 전체가 아니라 **받침 위치에 집중**돼 있다. `ink_keep≈1.0` 인데 missing≈extra 이므로
획이 증발한다기보다 **자리가 밀린다**. 최악 종성: ㅊ 0.129 > ㅀ 0.122 > ㄾ 0.121 > ㄶ 0.119 > ㅆ 0.100.
원본 영어 VAE 대비로는 이미 2.4~2.7배 개선된 상태(겹 0.265→0.100)지만 **유형 간 격차는 그대로 남았다**.
held-out 폰트가 오히려 약간 더 좋아 폰트 과적합은 아니다.

**② 생성 경로** (HTR 리더로 읽고 음절 정렬 후 자모 분해. 종성 유형 균등 텍스트, 유형별 450음절):

| 종성 유형 | 음절정확도↑ (GT렌더 / VAE복원 / **생성**) | 종성삭제율↓ | 종성오치환↓ |
|---|---|---|---|
| 없음 | 0.947 / 0.924 / **0.729** | — | — |
| 홑받침 | 0.976 / 0.900 / **0.711** | 0.000 / 0.007 / **0.042** | 0.011 / 0.053 / **0.140** |
| 쌍받침 ㄲㅆ | 0.969 / 0.913 / **0.720** | 0.002 / 0.004 / **0.056** | 0.018 / 0.044 / **0.149** |
| 겹받침 ㄳㄵ… | 0.956 / 0.844 / **0.493** | 0.000 / 0.013 / **0.058** | 0.036 / 0.118 / **0.404** |

- 진짜 취약한 건 ㄲ·ㅆ 가 아니라 **겹받침**이다. 쌍받침 음절정확도(0.720)는 홑받침(0.711)과 동급인데,
  **겹받침만 0.493 으로 22~24%p 낮다.**
- 지배적 실패 모드는 삭제(5.8%)가 아니라 **오치환 40.4%**(홑받침의 2.9배) — 두 자음 중 하나를 잃는 형태.
- 초성·중성 오류는 유형 무관하게 13~19%로 평탄 → 추가 손실은 **종성에만 국한**.
- **책임 분해**(겹받침 오치환): 리더바닥 3.6% → VAE 11.8% → 생성 40.4%. VAE 가 8%p, **T5 가 29%p**.
  VAE 재구성이 하드상한인 건 맞지만 겹받침에 한해선 **손실의 3/4 이 생성 단계**에서 생긴다
  → VAE 만 더 키워도 이 문제는 대부분 남는다.

> 측정 조건 주의: ② 는 종성 유형 표본을 맞추려고 **무작위 음절 나열**(OOD)을 쓴 통제 비교다.
> 절대 수준은 실제 문장보다 나쁘며, 읽어야 할 것은 **유형 간 상대 차이**다.
