#!/usr/bin/env bash
# 추론 진입점. 생성·시각화·릴리즈를 여기 하나로 부른다.
#
#   ./inference.sh best ["생성할 문장"]   best 모델로 생성 그리드 (가장 빠른 눈 확인)
#   ./inference.sh show   [args...]       결과 뷰어 [스타일ref | 정답 | 생성 cfg…]
#   ./inference.sh matrix [args...]       스타일 × 텍스트 매트릭스 뷰어
#   ./inference.sh echo   [args...]       echo(style==gen) 모델 나란히 비교
#   ./inference.sh refset [args...]       style ref 세트 생성 (없으면 위 뷰어가 못 돈다)
#   ./inference.sh release vae|model [..] HuggingFace 발행 (기본 dry-run, --push 로 실제 업로드)
#
# 뒤에 붙인 인자는 그대로 넘어간다:  ./inference.sh show --ckpt <ckpt> --cfgs 1.0 2.0
# 각 하위 명령의 전체 인자는 `--help` 로:  ./inference.sh show --help
#
# 환경변수
#   GPU=2         쓸 GPU (기본 0)
#   DRY=1         실행하지 않고 커맨드만 출력
#   CKPT=...      best 외 다른 체크포인트로 `best` 를 돌릴 때
#   REFSET=...    style ref 세트 lines_json (기본 data/ref_set_clean/train_lines.json)
#   PY=...        python 실행기 (기본 ./.venv/bin/python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
GPU="${GPU:-0}"
CKPT="${CKPT:-finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth}"
REFSET="${REFSET:-data/ref_set_clean/train_lines.json}"
cd "$ROOT"

run() {
  echo "+ CUDA_VISIBLE_DEVICES=$GPU ${*/#$PY/python}" >&2
  [ -n "${DRY:-}" ] || CUDA_VISIBLE_DEVICES="$GPU" "$@"
}
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

need_refset() {
  [ -f "$REFSET" ] && return
  echo "style ref 세트가 없다: $REFSET" >&2
  echo "  → ./inference.sh refset --per-font-train 5 --per-font-val 0 --no-augment --out data/ref_set_clean" >&2
  exit 1
}

infer_best() {
  [ -e "$CKPT" ] || { echo "체크포인트가 없다: $CKPT (./train.sh best status 참고)" >&2; exit 1; }
  need_refset
  local text="${1:-오늘 날씨가 좋아서 친구와 공원에서 커피를 마셨다 2024년}"
  shift || true
  local out="_debug/best_show.png"
  run "$PY" infer/show.py --ckpt "$CKPT" --lines-json "$REFSET" \
      --seed-text "$text" --cfgs 1.0 1.5 2.0 --n-writers 4 --out "$out" "$@"
  echo "→ $out"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  best)    infer_best "$@" ;;
  show)    run "$PY" infer/show.py "$@" ;;
  matrix)  run "$PY" infer/matrix.py "$@" ;;
  echo)    run "$PY" infer/echo_compare.py "$@" ;;
  refset)  run "$PY" custom_datasets/korean_fontset.py "$@" ;;
  release) run "$PY" tools/hf_upload.py "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "알 수 없는 명령: $cmd" >&2; echo >&2; usage >&2; exit 1 ;;
esac
