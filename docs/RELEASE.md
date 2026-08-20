# HuggingFace 릴리즈 (2026-07)

최고 조합을 **repo 2개**로 발행한다. 업로드/변환은 `tools/hf_upload.py` 로 재현한다.
사용법(공개 API 로 생성하는 법)만 필요하면 [README 「HuggingFace 릴리즈」](../README.md#huggingface-릴리즈) 절로 충분하다.

| repo | 내용 | 왜 분리하나 |
|---|---|---|
| [`HERIUN/eruku_korean`](https://huggingface.co/HERIUN/eruku_korean) | T5 + proj + 특수토큰 (`model.safetensors`) + remote code(`modeling_eruku.py`) | 상위 진입점 |
| [`HERIUN/emuru_vae_korean`](https://huggingface.co/HERIUN/emuru_vae_korean) | 한글 적응 VAE (diffusers `AutoencoderKL`) | `modeling_eruku.py` 가 `AutoencoderKL.from_pretrained(config.vae_name_or_path)` 로 **VAE 가중치를 그 repo 에서 실제로 내려받기** 때문 |

**T5 는 별도 업로드가 필요 없다** — `T5Config.from_pretrained(config.t5_name_or_path)` 로 구조만
받고 가중치는 상위 repo 의 weight 파일에서 전부 덮인다. 반면 VAE 는 가중치를 원격에서 가져오므로,
`full` 로 학습해 latent 가 이동한 우리 VAE 를 **config 가 가리키게** 해야 조합이 맞는다
(원본 `blowing-up-groundhogs/emuru_vae` 를 가리킨 채 T5 만 갈면 생성이 무너진다).

## 발행 절차

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

## 발행본을 로컬 도구로 쓰기

`--ckpt` 는 로컬 `.pth` 경로와 **HF repo id** 를 둘 다 받는다 (`infer/show.py:load_model`).
파일이 없고 `owner/name` 꼴이면 `model.safetensors` 를 받아 로컬 `Emuru` 클래스에 얹는다.

```bash
.venv/bin/python eval/htr_cer.py --ckpt HERIUN/eruku_korean --n 100 --coherent
.venv/bin/python infer/show.py   --ckpt HERIUN/eruku_korean --seed-text "한국어 손글씨"
./eval.sh best            # 로컬 ckpt 가 없으면 발행본으로 자동 폴백
```

성립하는 이유는 두 state_dict 가 **`t5_to_ocr.*` 하나만 빼고 완전히 같기** 때문이다
(실측: 773키 전부 bit-identical, `t5_to_ocr` 는 `alpha=1.0` 이라 미사용). 페어 한글 VAE 는
config 의 `vae_name_or_path` 를 읽어 자동으로 붙는다.

> ★ 아래의 style prefix 크롭 규약 차이는 릴리즈 **클래스**(`generate_handwriting` 이 내부에서
> 자르는 동작)의 문제라 이 경로에는 해당하지 않는다 — 가중치만 가져오고 생성·크롭은 전부
> 로컬 규약(`crop_gen`, special 첫 SOG 기준)으로 돈다. 이중 크롭 위험은 릴리즈 클래스를
> 직접 쓸 때만 생긴다.

## ⚠️ 릴리즈 클래스와 학습 클래스는 style prefix 규약이 다르다

릴리즈 `get_model_inputs` 는 style latent 뒤에 SOG placeholder 열(ones) 1개를 붙이고,
`generate` 가 반환 전에 style prefix 를 **내부에서 잘라낸다**. 로컬은 안 붙이고 안 자르며
`infer.show.gen_arr` 가 special 시퀀스의 첫 SOG 로 자른다. 둘을 섞어(릴리즈 출력에 gen_arr
크롭을 겹쳐) 쓰면 이중 크롭으로 생성물이 날아간다 — 릴리즈 경로는 `generate_handwriting`
하나로만 쓸 것. 검증은 `--verify-cer`(공개 API 그대로 CER 측정)로 한다.

### (2026-08 수정) 그 내부 크롭이 `style폭+1` 고정 오프셋이라 style echo 가 남았다

`style_text` 를 주면 모델이 SOG 를 낼 때까지 style 을 이어 그리는데(README 「`style_text`」 경고),
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

## ⚠️ 생성은 run-to-run 재현이 안 된다

같은 모델·같은 입력 2회에 픽셀 maxdiff 221, `--verify-cer`(n=24) 3회에 CER 0.035 / 0.031 / 0.074.
비결정적 GPU 커널이 argmax 를 뒤집으면 그 뒤 생성이 통째로 갈라지기 때문.
**n=24 는 게이트용이지 보고용 수치가 아니다** — 비교는 n≥100 페어링으로 할 것.

## ink bbox 크롭

업로드하는 `modeling_eruku.py` 에는 **ink bbox 크롭**(`generate_handwriting(bbox_crop=True)`)을
추가했다. 실제 스캔 손글씨의 여백이 latent std 를 폰트의 6~29배로 튀겨 runaway/blank 를
만드는 문제(→ [`VAE_ROBUSTNESS.md`](VAE_ROBUSTNESS.md) ②)의 추론측 해결책이다.
