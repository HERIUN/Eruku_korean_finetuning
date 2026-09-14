# assets/corpus — 출처와 규약

| 파일 | 용도 | 출처 · 라이선스 |
|---|---|---|
| `korean_lines.txt` | **학습** ko 어절 풀 (`paths.corpus_korean`) | ① 원본 941줄: 라인 이미지 데이터셋의 라벨 컬럼(출처 미확인, `docs/DATA_LIMITATIONS.md` D10) ② 2026-09-04 추가분: **한국어 위키백과 (CC BY-SA 4.0)** — HF `heegyu/kowiki-sentences` (`wiki_20221001_sents.parquet`) 에서 `tools/fetch_corpus.py` 로 추출 |
| `eval_lines_ko.txt` | **평가** 텍스트 (`eval/echo_metrics.py:EVAL_LINES_KO`) | `korean_lines.txt` 의 2026-09-04 스냅샷(중복 제거 후 941줄). **고정** — 학습 코퍼스가 커져도 CER 이 비교 가능하도록 분리했다 |
| `chars.txt` | **알파벳** 공급 (`custom_datasets/korean/alphabet.py`) | 한글 음절 2,345자. 샘플러는 쓰지 않는다(D2) |
| `english_words.txt` | en 단어 풀 | dwyl/english-words |

## 손대기 전에 알아야 할 것

- **알파벳은 `assets/korean_charset.json` 으로 고정돼 있다.** `load_charset()` 이 이 json 이 있으면
  코퍼스에서 재유도하지 않는다. HTR/VAE ckpt 이 `alphabet_size=2512` 에 핀 고정이라 이 값이 바뀌면
  기존 체크포인트가 전부 못 맞는다. `tools/fetch_corpus.py` 는 그래서 **고정 charset ∩ 폰트 교집합**
  밖의 글자가 하나라도 있는 줄을 통째로 버린다.
- **`korean_lines.txt` 를 바꿔도 평가 텍스트는 안 바뀐다** (`eval_lines_ko.txt` 분리). 반대로
  `eval_lines_ko.txt` 를 바꾸면 **과거 CER 수치와 비교가 끊긴다** — 바꾸지 말 것.
- 중복 라인은 `build_pools` 가 `set` 으로 접으므로 학습에 무해하지만, 파일 크기와 실제 정보량이
  어긋나 판단을 흐린다. 2026-09-04 에 22,996 → 941 로 정리했다.

## 위키백과 저작자 표시

추가된 한국어 문장은 [한국어 위키백과](https://ko.wikipedia.org) 기여자들의 저작물이며
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 을 따른다.
