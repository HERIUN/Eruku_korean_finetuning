"""이 저장소가 추가한 한글 데이터 경로. upstream 을 import 해 확장한다(역방향 금지).

| 모듈 | 역할 | upstream 대응 |
|---|---|---|
| `split.py`    | `KoreanSplitFontSquare` — 온라인 style/gen split 데이터셋 | `upstream.font_square.font_square_split.OnlineSplitFontSquare` 서브클래스 |
| `fontset.py`  | `MixedLineSampler`(한글/영어/숫자/랜덤음절 혼합 라인) + 오프라인 생성기 | `upstream.font_square.font_square.TextSampler` 대체 |
| `alphabet.py` | 한글 charset/Alphabet | `upstream.alphabet.Alphabet` 확장 |
| `aux.py`      | 보조 HTR 데이터셋(VAE 한글적응용) | 없음(신규) |
| `handb.py`    | HanDB 실제 손글씨 데이터셋 | `upstream.real_datasets` 규약만 차용 |

무엇을 왜 바꿨는지는 [`docs/DATA_GENERATION.md`](../../docs/DATA_GENERATION.md) §8,
알려진 한계는 [`docs/DATA_LIMITATIONS.md`](../../docs/DATA_LIMITATIONS.md) 참고.
"""
