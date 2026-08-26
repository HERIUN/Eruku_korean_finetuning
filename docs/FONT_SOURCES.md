# 폰트 소스와 수집 (2026-08)

학습 writer 1명 = 폰트 1종이라 **writer 수를 늘리는 유일한 방법이 폰트 확보**다
(README 「실제 사람 손글씨 재현도가 낮다」의 원인). 이 문서는 어디서 무엇을 어떻게 받았는지,
그리고 다시 받으려면 뭘 치면 되는지를 적는다.

| 소스 | 수집기 | 결과 | 위치 |
|---|---|---|---|
| Google Fonts (korean subset) | `assets/download_script/tools_fetch_gf.py --lang ko` | 기존 학습 폰트의 일부 | `assets/fonts_korean_v2/` |
| Google Fonts (latin, Handwriting/Display) | `tools_fetch_gf.py --lang en` | 177종 | `assets/fonts_english/` |
| **눈누 noonnu.cc** | `tools_fetch_free.py --src noonnu` | **997종** (+ 별도 격리 32) | `assets/fonts_korean_new/noonnu/` |
| **온글잎 ownglyph.com** | `tools_fetch_free.py --src ownglyph` | **995종** | `assets/fonts_korean_new/ownglyph/` |

`assets/fonts_korean_new/` 는 **수집 staging** 이다 — `.gitignore` 로 커밋에서 빠지고,
학습 파이프라인(`fonts_korean_v2`, `fonts_charsets.json`, `MixedLineSampler`)은 아직 이걸 안 본다.

---

## 1. 눈누 (noonnu.cc) — 상업용 무료 한글 폰트 모음

### 왜 webfont 를 받나

눈누는 폰트 파일을 자기가 호스팅하지 않는다. 페이지의 다운로드 버튼은 **제조사 사이트로 나간다**
(`design.godo.co.kr`, `earlyfont.com`, …). 제조사마다 폼·약관 동의·zip 구조가 전부 달라 자동화가 안 된다.

대신 페이지에 미리보기용 **webfont 가 `@font-face` 로 박혀 있다**:

```css
@font-face { font-family: 'EF_jejudoldam';
  src: url('https://cdn.jsdelivr.net/gh/projectnoonnu/noonfonts_2210-EF@1.0/EF_jejudoldam.woff2'); }
```

이게 유일한 자동 경로다. **서브셋이 아니라 완본**임을 확인했다 — woff2 헤더의 `totalSfntSize` 가
2,546,780 바이트(압축본은 220KB)로, 한글 완성형 전체가 들어 있다.

> 폰트 페이지 본문의 `@font-face` 는 그 폰트 전용이다. 사이트 UI 폰트(Pretendard·Paperlogy 등)는
> 외부 CSS 로 따로 로드되므로 섞여 들어오지 않는다.

### 목록 확보

`https://noonnu.cc/sitemap.xml` → `font_page/<id>` **1,172개**. robots.txt 는 `/font_page` 를 막지 않는다.
(`/index?page=N` 은 24개씩 무작위 순서라 전수 수집에 부적합)

### 페이지에서 뽑는 것

| 항목 | 추출 방식 |
|---|---|
| 폰트명(한글) | `<title>` 의 `\|` 앞부분 |
| webfont URL | `@font-face` 블록의 `url(...)`. `//cdn…` 프로토콜 상대경로 → `https:` 보정 |
| 라이선스 요약표 | `라이선스 요약표` 이후 `<tr>` → `{인쇄·웹사이트·포장지·영상·임베딩·BI/CI·OFL: 허용여부}` |
| 제조사 다운로드 URL | `class="noon-yellow-button"` 앵커의 `href` (1,171/1,172 확보) |

전부 `_catalog_noonnu.json` 에 페이지 단위로 저장한다. **폰트 파일보다 이 카탈로그가 더 오래 쓸모 있다** —
라이선스 판단과 원본 재취득에 필요한 정보가 여기 다 있다.

### 예외 경로 두 가지

1. **CSS import** — 구형 폰트는 `@font-face` 대신 제조사 CSS 를 import 한다
   (`//cdn.jsdelivr.net/font-iropke-batang/1.2/font-iropke-batang.css`). CSS 를 한 단계 더 따라가
   그 안의 `@font-face` 를 읽는다. 안 따라가면 `NOWEBFONT` 로 빠진다.
2. **URL 에 한글** — `…/행복고흥M.woff`, `…/62570체.woff` 처럼 경로에 한글이 든 폰트가 있다.
   `urllib.parse.quote` 로 인코딩해야 받아진다 (안 하면 전부 `DLFAIL`).

### 굵기 선택

한 페이지에 여러 굵기가 걸려 있으면 기본은 **1종만** 받는다. 정렬 키(`weight_rank`)는
`regular` 포함 → Bold/Black/Thin 류 아님 → 파일명이 `M`/`R`/`Rg`/`Medium` 으로 끝남 → 짧은 이름 순.
`GodoM`(Medium) 과 `GodoB`(Bold) 중 `GodoM` 을 고르라고 만든 규칙이다. 전부 받으려면 `--all-weights`.

### 라이선스 (수집분 1,032건 기준)

| 임베딩 항목 | 폰트 수 |
|---|---|
| 사용 가능 | 899 |
| **조건부 허용** | **88** |
| 표기 없음 | 45 |

OFL 표기가 있는 폰트는 0건이다. ⚠️ **수집 = 사용 허가가 아니다.** 학습 투입 전에
`_catalog_noonnu.json` 의 `license` 를 보고 판단할 것.

---

## 2. 온글잎 (ownglyph.com) — 개인 손글씨 폰트

### 목록: 공개 Firestore

사이트가 Next.js + Firebase 라 폰트 목록이 **공개 Firestore 컬렉션**에 그대로 있다.
`__NEXT_DATA__` 는 비어 있고 클라이언트가 `fonts/fontMenu/ownglyphFontV2` 를 읽는 구조라,
같은 REST 엔드포인트를 그대로 쓴다 (`v6x-ownglyph` / 웹 apiKey 는 JS 번들에 공개된 값).

`fontUrl` 이 **S3 TTF 직링크**라 변환이 필요 없다. 경로에 한글 폰트명이 들어 있어 `quote` 는 필요하다.

배포 조건은 `allowOwnglyphDeploy && completed && !canceled` — 문서 12,097건 중 **11,989종**이 해당하고
전부 `fontStyle == 손글씨` 다.

### ⚠️ 개인정보

이 컬렉션의 원본 문서에는 폰트 주인의 **이메일·전화번호·생년월일·실명**(`notiEmail`,
`notiPhoneNumber`, `copyrightOwnerBirthDate`, …)이 그대로 들어 있다.

**반드시 `mask.fieldPaths` 로 폰트 필드만 요청한다.** 받아놓고 나중에 거르는 방식은 쓰지 않는다.
수집기가 요청하는 9개 필드가 전부다:

```
fontUID, fontNameKor, fontNameEng, fontUrl, fontStyle, fontIndex,
allowOwnglyphDeploy, completed, canceled
```

`_catalog_ownglyph.json` 도 이 9개만 담는다 (검증: 이메일 패턴 0건, 전화 패턴은 UUID 오탐만).

### 샘플링

11,989종 전량은 ~17GB 라 **`fontIndex` 정렬 후 균등 stride 로 1,000종**만 받았다(난수 아님 → 재현됨).
카탈로그에는 **11,989종 전량 메타**가 있으므로 더 필요하면 Firestore 재조회 없이 이어받을 수 있다.

### 라이선스

⚠️ 온글잎 폰트는 **실명 개인이 자기 손글씨를 폰트화한 것**이다. 무료 배포이긴 하나,
이걸로 손글씨 **생성 모델**을 학습시키는 건 수집과는 별개의 판단이 필요하다.

---

## 3. 중복 제거

기존 보유분(`fonts_korean_v2/train`·`test`, `custom_hw_fonts`, `fonts_english`)과 **staging 디렉토리 자신**을
인덱싱해 2단으로 거른다. staging 을 포함하는 덕에 재실행 이어받기와 **소스 간 교차 중복**이 같이 해결된다
(눈누가 온글잎 폰트를 일부 싣고 있어 실제로 5종이 여기서 걸렸다).

| 단 | 키 | 잡는 것 |
|---|---|---|
| ① 이름 | `tools_fetch_gf.norm(stem)` (영숫자만 남기고 소문자화) | `Cafe24Dongdong` 류 정확 일치 |
| ② 글리프 | 대표 22자 아웃라인을 upm 정규화 후 `sha1` (`DecomposingRecordingPen`) | 이름이 다른 같은 폰트 |

②가 없으면 놓쳤을 실제 사례: `GowunBatang-Regular`↔`GowunBatang`, `Dongle-Regular`↔`Dongle`,
`116watermelon`↔`116Watermelon`, `DOSIyagiMedium`↔`DOSIyagi`, `ARCHISCULPTURE_v200`↔`ARCHISCULPTURE`,
`Dovemayo_wild`↔`DovemayoWild`. 눈누 내부 중복(`Mona10`/`Mona12`/`Mona10x12` 가 같은 파일)도 잡힌다.

**총 84종**이 중복으로 제외됐다 (noonnu 79 + ownglyph 5).

---

## 4. 사용법

```bash
# 눈누 전량 (sitemap 1,172 페이지)
.venv/bin/python assets/download_script/tools_fetch_free.py --src noonnu --workers 6 --sleep 0.5

# 온글잎 균등 샘플 1,000종
.venv/bin/python assets/download_script/tools_fetch_free.py --src ownglyph --n 1000 --workers 12

# 온글잎 목록만 갱신 (파일 안 받음)
.venv/bin/python assets/download_script/tools_fetch_free.py --src ownglyph --catalog-only

# 실패분만 재시도 (성공분은 이전 리포트에서 승계, 카탈로그는 병합)
.venv/bin/python assets/download_script/tools_fetch_free.py --src noonnu --workers 4 \
  --retry assets/fonts_korean_new/_fetch_report_noonnu.json
```

주요 인자: `--min-cov 0.95`(KS 표본 커버리지 하한) `--limit`(디버그) `--all-weights` `--out`

### ⚠️ 워커 수 — 스레드가 아니라 프로세스, 그리고 6 이하

- **woff2 → sfnt 재컴파일이 폰트당 ~2.8초 CPU** 다. 스레드 풀은 GIL 때문에 전혀 안 빨라진다
  (실측: 8스레드 = 순차와 동일 처리량). 그래서 `ProcessPoolExecutor` 를 쓴다.
  온글잎은 이미 평문 TTF 라 재컴파일 자체를 건너뛴다.
- **눈누는 동시 요청에 민감하다.** 16워커로 돌렸더니 페이지 요청의 **22%가 PAGEFAIL** 로 떨어졌다.
  6워커 + `--sleep 0.5` 에서 1,172페이지 PAGEFAIL 0. jsdelivr(폰트 CDN)는 안 막으므로 병목은 눈누 쪽이다.

### 재시도가 필요한 이유

`--retry` 없이 다시 돌리면 1,172페이지를 또 긁으면서 새 PAGEFAIL 을 만들고,
**카탈로그를 덮어써 이전 결과보다 나빠질 수 있다**. 실패분 재시도는 반드시 `--retry` 로.

---

## 5. 산출물

```
assets/fonts_korean_new/          # .gitignore 됨, 4.1GB
├── noonnu/                         997종  ← 학습 투입 가능
├── ownglyph/                       995종  ← 학습 투입 가능
├── _partial/                        28종  알파벳 2,345 미달 (§7) + REASON.txt
├── _rejected/                        4종  렌더 불가·빈 글리프 (§6) + REASON.txt
├── _catalog_noonnu.json          1,172행: 폰트명·페이지·webfont URL·라이선스 요약표·제조사 URL
├── _catalog_ownglyph.json       11,989행: 배포가능 전량 메타(PII 제외)
├── _fetch_report_noonnu.json     폰트별 status·coverage·space_glyph
└── _fetch_report_ownglyph.json
```

리포트 `status` 값: `OK` / `SKIP-EXIST`(이미 있음) / `SKIP-DUP`(중복) / `LOWCOV`(한글 커버리지 미달) /
`NOWEBFONT` / `DLFAIL` / `SAVEFAIL` / `BADFONT` / `PAGEFAIL`

---

## 6. 검증 (실측)

```bash
# 폰트 품질 감사 (두부 글리프 · malformed 기하)
.venv/bin/python tools/font_audit.py --fonts-dir $PWD/assets/fonts_korean_new/noonnu \
  --out _val/font_audit_noonnu.txt        # ⚠️ --fonts-dir 는 절대경로

# 렌더 가능성 (font_audit 이 폰트 하나에서 죽은 적이 있어 별도 확인)
# PIL ImageFont.truetype(...).getbbox("가힣A0 ") 를 전 폰트에 시도
```

| | noonnu | ownglyph |
|---|---|---|
| 감사 대상 | 1,027 | 995 |
| 깨끗(코퍼스 100%·비정상 0) | 992 | **995 (전부)** |
| 코퍼스 글자 미지원 | 30 | 0 |
| 비정상 기하 글리프 | 6 | 0 |
| 렌더 실패 | 2 → 격리 | 0 |

온글잎은 감사를 전부 통과했다 — 실제 사람 손글씨 995종이 무결하게 들어왔다는 뜻이고,
writer 다양성 관점에서 이번 수집의 핵심이다.

### 알려진 함정 3가지

1. **cmap 은 있는데 글리프가 빈 폰트** — `EVECondom-Regular`, `KIMB_ENG-BOLD` 는 KS 커버리지 검사를
   통과했지만 한글 글리프가 전부 비어 있었다(감사에서 2,345/2,345 미지원). cmap 만 믿으면 안 된다. → 격리.
2. **FreeType 이 못 여는 폰트** — `Fairytale_ddobak`, `Hana_handwriting` 이 `OSError: too many
   function definitions` 로 `font_audit.py` 를 죽였다. → 격리 (`_rejected/REASON.txt`).
3. **공백(U+0020) 글리프가 없는 폰트 23종** — 띄어쓰기가 두부로 렌더된다. 버리진 않고
   리포트에 `space_glyph: false` 로 표시했다. 라인 렌더에 쓰기 전에 걸러야 한다.

`--min-cov 0.95` 에 걸린 `LOWCOV` 40종은 눈누가 **웹폰트를 유니코드 서브셋으로 잘라 올린** 것들이다
(`NotoSansKR`, `NanumGothic`, `NanumMyeongjo`, `NanumPenScript` …). 필터가 의도대로 동작한 것이고,
원본 전체 폰트가 필요하면 `tools_fetch_gf.py --lang ko` 로 Google Fonts 에서 받으면 된다.

미수집 `DLFAIL` 5종(KoPub바탕, 제주고딕/명조/한라산, 한나체)은 폐지된
`fonts.gstatic.com/ea/`(Google Fonts Early Access) 를 가리켜 CDN 에 파일이 없다. 역시 Google Fonts 경로로.

---

## 7. 오버뷰 · 실렌더 검사 (2026-08-24)

`render_overview.py` 는 폴더당 PNG 1장이라 1,025종이면 11,640px 짜리가 나와 못 본다.
100종 배치로 쪼개 병렬 렌더했다 → `_val/overview_new/{noonnu,ownglyph}_NN.png` 21장.

이어서 전 폰트를 **실제로 래스터**해 판정했다(`_val/font_scan_new.json`).

> ⚠️ **기준은 `korean_lines.txt` 가 아니라 학습 알파벳 2,345 음절이다**
> (`custom_datasets/korean/alphabet.py:26` — `korean_lines.txt` + **`chars.txt`**, 후자가 대부분을 공급).
> 샘플러도 코퍼스 어절만 뽑는 게 아니라 **폰트 교집합 전 음절에서 rand 토큰**을 만든다
> (`fontset.py` D2 주석). `korean_lines.txt` 의 754자로 재면 문제 폰트를 6종으로 과소평가한다.

| | noonnu | ownglyph |
|---|---|---|
| 알파벳 2,345 음절을 못 그림 | **28종** | **0종** |
| 한글 placeholder(두부) | 6종 (각 ~8,400자) | 0 |
| 공백(U+0020) 글리프 없음 | 38종 | 0 |
| 풀 전체 한글 교집합 | 496자 → **28종 제외 후 2,350자** | **2,350자** |

**온글잎 995종은 손댈 게 없다** — 전부 정확히 KS X 1001 2,350자를 그리고 공백도 있다. 그대로 드롭인 가능.

### 제외 (`assets/fonts_korean_new/_partial/`, 되돌리려면 `noonnu/` 로 옮기면 됨)

`charset_rule=intersection` 풀에 섞이면 **교집합을 496자까지 끌어내려** 학습 charset 이 무너진다.
28종 중 2종이 피해의 대부분이다:

| 폰트 | 부족 | |
|---|---|---|
| `establishThornFontOTF` | 1,845자 | 이 1종만 빼도 496 → 1,695 |
| `establishRoomNo703OTF` | 594자 | 2종 빼면 2,130 |
| `116angmuburi` · `HSSanTokki20` · `snByulByul` · `116angduk_honesty15` | 50~79자 | |
| 나머지 22종 | 1~11자 | |

### 두부는 왜 안 지웠나

`EliceDigitalBaeum` · `HiKR-ExtraBold` · `BnviitLasik` · `SLEIGothicTTF` · `SacheonHangGong` ·
`SacheonUju` 6종은 cmap 에 11,172 음절을 다 선언해놓고 실제로는 2,350~2,780 만 그리고
나머지 ~8,400자를 ▨ ▣ ⊠ 같은 placeholder 로 채웠다(`_val/problem_fonts.png` 에서 육안 확인).

그런데 **실제로 그리는 집합이 학습 알파벳 2,345 를 다 덮는다** → 폰트를 버릴 이유가 없다.
문제는 폰트가 아니라 charset 쪽이다 (아래).

> 두부 판정은 **한글 음절에만** 적용해야 한다. `GabiaSai` · `DeltaDotumKR` 은 À Á Â 가 전부
> 민짜 `A` 로 그려져 원글자까지 "비트맵 공유"로 잡히는 **오탐**이었다 — 실제로는 정상 렌더된다.

### ⚠️ charset 캐시가 cmap 만 본다 (D6)

`custom_datasets/korean/split.py:206` 이 `chr(c) for c in get_cmap(p)` 로 charset 을 만든다.
그래서 ① **빈 글리프**(cmap 엔 있는데 잉크 없음 → 이미지엔 안 그려지는데 라벨엔 글자가 남는다 =
**라벨 불일치**) ② **두부 placeholder** 를 원리상 못 거른다. 같은 함수의 docstring 이 경고하는
D6 가 바로 이것이고, cmap 기반으로는 못 고친다.

→ **`tools/font_charset.py`** 가 실제로 래스터해서 charset 을 만든다(같은 포맷, 드롭인 교체).

```bash
.venv/bin/python tools/font_charset.py --fonts-dir assets/fonts_korean_new/ownglyph
```

판정 3단계: ① 공백·제어(카테고리 Z/C)는 무조건 통과 — 잉크 없는 게 정상이라 잉크 검사에 걸리면
안 된다 ② `getmask(ch).getbbox()` 가 None 이면 **빈 글리프** 로 제외 ③ 렌더 비트맵을 해시해
같은 값을 `--tofu-min`(기본 10)자 이상이 공유하면 **placeholder** 로 제외.

> ⚠️ **ASCII 인쇄가능(0x20~0x7E)은 ③에서 면제**한다. À Á Â 가 민짜 `A` 로 그려지는 폰트에서
> 비트맵 공유가 임계값을 넘는데, 잘못 그려진 쪽은 **À 지 `A` 가 아니다**
> (GabiaSai-Regular 에서 A E I O U a e i 가 오탐으로 잡혔다).
> 반대로 희귀 영역(한글 음절·한자·기호·악센트)의 대량 공유는 전부 진짜 두부였다 —
> KBIZHanmaumMyungjo 의 한자·기호 1,555자, TerrarumSansBitmap 의 제어문자 54자 등.

한계: 글자마다 **다르게** 깨진 글리프는 비트맵이 서로 달라 못 잡는다 → `tools/font_audit.py`
기하 검사가 그 영역을 맡는다.

### 공백 없는 38종

`render_font.py:75` 가 charset 에 없는 글자를 **라벨에서도 같이 지우므로** 불일치는 안 난다.
다만 그 writer 들은 띄어쓰기를 한 번도 보여주지 않는다(2~3어절 학습에선 손해).
버리진 않았고 `_val/font_scan_new.json` 의 `space_ok:false` 로 걸러 쓸 것.

---

## 8. 아직 안 한 것

- **학습 투입** — `fonts_korean_v2/train` 이동, `fonts_charsets.json` 갱신,
  `MixedLineSampler` 비중 재조정, README/`docs/DATA_GENERATION.md` 폰트 수 갱신.
- **라이선스 적합성 최종 판단** (1·2절 ⚠️).
- **수집 폰트의 재배포** — HF 업로드 등은 하지 않는다.
