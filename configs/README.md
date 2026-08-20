# configs — 데이터 생성부터 학습까지 한 곳에서

CLI 인자만 있으면 "무엇을 조절할 수 있는지"가 스크립트 여기저기에 흩어진다.
그래서 **조절 가능한 전부를 [`base.yaml`](base.yaml) 한 파일에 주석과 함께** 두고,
각 레시피는 거기서 달라지는 것만 적는다.

| 파일 | 무엇 |
|---|---|
| [`base.yaml`](base.yaml) | **전체 파라미터 인벤토리 + 기본값**. 새 항목은 여기 먼저 추가 |
| [`phase1.yaml`](phase1.yaml) | 논문 Phase 1 — 짧은 2~3어절로 glyph 학습 (65000 step) |
| [`phase2.yaml`](phase2.yaml) | 논문 Phase 2 — Phase 1 에서 resume, 긴 줄 +5000 step |
| [`korean_from_en.yaml`](korean_from_en.yaml) | 영어 pretrained → 바로 한글 (Phase 1 생략, 실사용) |

## 우선순위

```
CLI  >  yaml  >  코드 기본값(make_parser / SAMPLER_DEFAULTS)
```

```bash
python train/phase1.py                            # configs/phase1.yaml
python train/phase1.py --lr 3e-5 --batch-size 64  # 두 개만 CLI 로 덮어쓰기
python train/phase1.py --config configs/my.yaml   # 다른 config
```

## 상속

`_base_: base.yaml` (경로는 그 yaml 기준 상대). **깊은 병합**이라 자식은 바꿀 것만 적으면 된다.

```yaml
_base_: base.yaml
data:
  sampler:
    punct_prob: 0.0        # 나머지 sampler 항목은 base 값 유지
```

## 구조

섹션(`run` / `model` / `optim` / `data` / `val`)은 **사람이 읽기 위한 그룹**일 뿐이다.
어느 섹션에 두든 동작은 같고, 키 이름이 CLI 인자명(`--style-words` → `style_words`)과
같기만 하면 된다. 정의되지 않은 키는 **에러**로 막는다(오타가 조용히 무시되지 않게).

예외는 `data.sampler` 하나다. 이건 argparse 인자가 아니라 텍스트 샘플러
(`custom_datasets/korean/fontset.py:SAMPLER_DEFAULTS`)에 dict 통째로 전달된다.

경로 값(`english_fonts_dir`, `resume`, `eruku_pretrained` …)은 상대로 적어도
저장소 루트 기준 절대경로로 펴진다 → cwd 와 무관하게 같은 곳을 가리킨다.

## 새 파라미터 추가하기

1. `train/core.py:make_parser()` 에 `add_argument` (코드 기본값 = 최종 fallback)
2. [`base.yaml`](base.yaml) 의 알맞은 섹션에 같은 이름으로 + 한 줄 설명
3. 샘플러 파라미터라면 대신 `custom_datasets/korean/fontset.py:SAMPLER_DEFAULTS` 와
   `base.yaml` 의 `data.sampler` 두 곳

## 실제로 적용된 설정 확인

run 디렉토리의 `train_config.yml` 에 run 별로 누적 기록된다 —
CLI 덮어쓰기까지 반영된 **최종 값**과 `sampler` 전체가 함께 남으므로, 나중에
"이 체크포인트는 어떤 데이터로 학습됐나"를 그 파일 하나로 되짚을 수 있다.
resume 인데 이전 run 과 값이 다르면 시작 시 `[config WARN]` 이 뜬다.
