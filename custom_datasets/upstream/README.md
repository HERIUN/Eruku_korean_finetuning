# upstream — Eruku 원본 vendored 복사본

출처: <https://github.com/Blowing-Up-Groundhogs/Eruku> (초기 커밋 `fe688c8` 에 통째로 들여옴)

## 규칙

1. **여기는 고치지 않는다.** 원본과 diff 를 뜰 수 있어야 upstream 변경을 따라갈 수 있다.
2. 한글용 변경은 `custom_datasets/korean/` 에 **서브클래스·래퍼**로 넣는다.
3. 부득이하게 고쳤다면 아래 표에 반드시 기록한다.
4. 의존 방향은 **korean → upstream 단방향**. upstream 이 korean 을 import 하면 안 된다.

## 원본을 고친 부분 (전부)

| 파일 | 변경 | 커밋 | 이유 |
|---|---|---|---|
| `font_square/font_square_split.py` | `RandomWarping(grid_shape=(5,2), p=1.0)` → `p=0.5` | `e99bc43` | p=1.0 이면 휨 없는 샘플이 0개라 추론도 항상 휘어 나옴. 원본 기본값 복원 |

그 외 파일은 원본 그대로다.

## 구성

```
upstream/
├─ alphabet.py          문자 ↔ 정수 인코더
├─ constants.py         특수토큰(PAD/SOS/EOS), FONT_SQUARE_CHARSET
├─ subsequent_mask.py   디코더 causal mask
├─ font_square/         온라인 합성 데이터 파이프라인 (핵심)
│  ├─ render_font.py         Render: 폰트 1개 → 텍스트 이미지 (크기 캘리브레이션)
│  ├─ font_square.py         OnlineFontSquare(단일 텍스트) + TextSampler(nltk 영어)
│  ├─ font_square_split.py   OnlineSplitFontSquare(style+gen) ← 한글판이 상속하는 클래스
│  ├─ font_transforms*.py    증강 transform 모음
│  └─ tps/                   Thin-Plate-Spline 워핑 (한글 경로에서는 미사용)
└─ real_datasets/       원본 실제-데이터 로더 (한글 HanDB 는 korean/handb.py)
```

## 한글 경로가 실제로 쓰는 것 / 안 쓰는 것

| upstream 요소 | 한글 경로에서 | 비고 |
|---|---|---|
| `Render`, `make_renderers` | **사용** | 폰트 로딩·캘리브레이션·charset 필터 |
| `OnlineSplitFontSquare` | **상속** | `korean.split.KoreanSplitFontSquare` |
| `font_transforms_split` 의 개별 transform | **사용**(직접 조합) | 원본 `Compose` 대신 단계별 명시 호출 |
| `TextSampler`(nltk 6개 코퍼스) | **미사용** | `korean.fontset.MixedLineSampler` 로 대체 |
| `RandomWarping`(TPS) | **미사용** | 한글 정책상 제거 (`--legacy-transform` 일 때만 탐) |
| `RandomInvert` | **미사용** | 곱셈 합성에서 무정보 |
| `tps/` C++ 확장 | **미사용** | warp 를 안 쓰므로 |

무엇을 왜 바꿨는지는 [`docs/DATA_GENERATION.md`](../../docs/DATA_GENERATION.md) §8-4 표에 정리돼 있다.
