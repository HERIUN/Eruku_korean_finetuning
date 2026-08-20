"""YAML 설정 로더 — 데이터 생성부터 학습까지 한 파일에서 관리한다.

우선순위는 **CLI > YAML > 코드 기본값**. 즉 config 로 전부 적어 두고, 실험 때만
`--lr 3e-5` 처럼 한두 개를 CLI 로 덮어쓴다.

    python train/phase1.py                                  # configs/phase1.yaml
    python train/phase1.py --config configs/my_run.yaml      # 다른 config
    python train/phase1.py --lr 3e-5 --batch-size 64         # 일부만 CLI 로 덮어쓰기

YAML 구조는 argparse dest 이름을 그대로 쓰되 **읽기 좋게 섹션으로 묶은 것**뿐이다.
섹션 이름 자체는 의미가 없고(어느 섹션에 두든 동작 동일) 사람이 훑어보기 위한 그룹이다.

    run:    { out, seed, device, resume, ... }
    model:  { t5_checkpoint, vae_checkpoint, eruku_pretrained, ... }
    optim:  { batch_size, grad_accum, lr, max_steps, ... }
    data:   { online_split, style_words, gen_words, english_frac, ...,
              sampler: { weights, punct_prob, ... } }   ← sampler 만 예외(dict 통째로 전달)
    val:    { val_every, val_fonts_dir, ... }

`_base_: other.yaml` 로 상속한다(상대경로 = 그 yaml 기준). 깊은 병합이라 자식은 필요한
키만 덮어쓰면 된다.

정의되지 않은 키는 **에러**다. 오타로 조용히 무시되는 것을 막기 위함이다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# data.sampler 만 argparse dest 가 아니라 dict 통째로 데이터 계층에 전달된다.
SAMPLER_KEY = "sampler"

REPO_ROOT = Path(__file__).resolve().parents[1]   # configs/ 의 부모 = 저장소 루트

# yaml 에는 `assets/fonts_english` 처럼 짧게 적고, 여기서 저장소 루트 기준 절대경로로 편다.
# (cwd 가 어디든 같은 곳을 가리키게. 이미 절대경로면 그대로 둔다.)
PATH_KEYS = frozenset({
    "english_fonts_dir", "val_fonts_dir", "val_english_fonts_dir",
    "korean_fonts_dir", "backgrounds_dir",
    "corpus_korean", "corpus_chars", "corpus_english",
    "lines_json", "resume", "ocr_checkpoint",
    # eval / infer
    "fonts_dir", "fonts_dir_ko", "fonts_dir_en", "htr_checkpoint", "style_image", "out",
    # eruku_pretrained / vae_checkpoint 는 제외 — HF repo id 도 받으므로 경로로 펴면 안 된다
})


def load_config(path) -> dict:
    """YAML 로드 + `_base_` 상속 해석(깊은 병합)."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"config 없음: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_cfg = load_config((path.parent / base).resolve())
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def flatten(cfg: dict, scopes=None) -> tuple[dict, dict | None]:
    """섹션 dict → (argparse dest 평면 dict, sampler dict).

    scopes=None  : 모든 섹션을 합친다. 섹션은 순전히 사람이 읽기 위한 그룹(학습 config).
    scopes=(...) : **그 섹션들만**, **주어진 순서대로** 쓴다(뒤가 앞을 덮어씀). 스크립트마다
                   인자 집합이 다른 eval/infer 에서, 한 파일에 공통(common) + 스크립트별
                   섹션을 두고 골라 쓰기 위한 것. 예: ("common", "echo") 면 echo 가 common 을 덮는다.
    """
    flat, sampler = {}, None
    items = ([(s, cfg[s]) for s in scopes if s in cfg] if scopes is not None
             else list(cfg.items()))
    for section, body in items:
        if not isinstance(body, dict):
            raise ValueError(f"config 최상위 '{section}' 은 섹션(dict)이어야 한다 (값: {body!r})")
        for k, v in body.items():
            if k == SAMPLER_KEY:
                sampler = v
                continue
            if k in flat and scopes is None:
                # 학습 config 는 섹션이 순전히 장식이라 같은 키가 두 번 나오면 실수다.
                # scopes 를 준 경우는 의도된 덮어쓰기(뒤 섹션 우선)이므로 허용.
                raise ValueError(f"config 키 중복: '{k}' (섹션 여러 곳에 존재)")
            flat[k] = _abs_path(v) if k in PATH_KEYS else v
    return flat, sampler


def _abs_path(v):
    """상대경로면 저장소 루트 기준 절대경로로. None/절대경로는 그대로."""
    if not isinstance(v, str) or not v:
        return v
    p = Path(v)
    return v if p.is_absolute() else str(REPO_ROOT / p)


def parse_args(parser: argparse.ArgumentParser, argv=None, default_config=None, scopes=None):
    """config 를 반영해 파싱한다. `args.sampler` / `args.config_path` 가 붙는다.

    scopes: eval/infer 처럼 한 config 를 여러 스크립트가 나눠 쓸 때 읽을 섹션들
            (예: ("common", "cer")). None 이면 전 섹션.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_config)
    known, _ = pre.parse_known_args(argv)

    sampler = None
    if known.config:
        flat, sampler = flatten(load_config(known.config), scopes=scopes)
        valid = {a.dest for a in parser._actions}
        unknown = sorted(set(flat) - valid)
        if unknown:
            raise ValueError(
                f"config 에 정의되지 않은 키: {unknown}\n"
                f"  (CLI 인자 이름과 같아야 한다. 예: --style-words → style_words)\n"
                f"  사용 가능: {sorted(valid - {'help'})}")
        parser.set_defaults(**flat)
        # config 가 값을 주면 required 인자도 CLI 에서 생략할 수 있어야 한다
        for a in parser._actions:
            if a.dest in flat and getattr(a, "required", False):
                a.required = False

    args = parser.parse_args(argv)
    args.sampler = sampler
    args.config_path = known.config
    return args


#: 없으면 학습이 곧바로 실패하는 입력 자산 (모델 ckpt 는 HF 자동 다운로드라 제외)
REQUIRED_PATHS = ("corpus_korean", "corpus_chars", "corpus_english",
                  "korean_fonts_dir", "backgrounds_dir", "english_fonts_dir")


def check_paths(args) -> list[str]:
    """존재하지 않는 입력 자산 경로 목록. 학습 시작 전에 찍어 주기 위한 것."""
    missing = []
    for k in REQUIRED_PATHS:
        v = getattr(args, k, None)
        if v and not Path(v).exists():
            missing.append(f"{k} = {v}")
    return missing


def effective(args, parser=None) -> dict:
    """run 기록용 — 실제로 적용된 전체 설정(dict)."""
    d = {k: v for k, v in vars(args).items() if k not in ("sampler",)}
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
    if getattr(args, "sampler", None) is not None:
        d["sampler"] = args.sampler
    return d
