#!/usr/bin/env bash
# 평가 진입점. 정량 지표·감사·일회성 실험을 여기 하나로 부른다.
#
#   ./eval.sh best [args...]      best 모델의 README 수치 재현 (한글/영어 CER, n=100)
#   ./eval.sh cer     [args...]   한글 HTR CER  ← 한글 주 지표 (렌더 바닥 ~0.00)
#   ./eval.sh easyocr [args...]   easyocr CER (한글은 saturated, 교차검증용)
#   ./eval.sh echo    [args...]   echo 정량 (MSE/SSIM/LPIPS/FID)
#   ./eval.sh all     [args...]   통합 배터리 (여러 모델 × ko/en 한 번에)
#   ./eval.sh fonts   [args...]   폰트 품질 감사 (두부·malformed 글리프)
#   ./eval.sh exp [이름] [args]   experiments/<이름>.py 실행 (이름 없으면 목록)
#
# 뒤에 붙인 인자는 그대로 넘어간다:  ./eval.sh cer --ckpt <ckpt> --n 200 --coherent
# 각 하위 명령의 전체 인자는 `--help` 로:  ./eval.sh cer --help
#
# 환경변수
#   GPU=2         쓸 GPU (기본 0)
#   DRY=1         실행하지 않고 커맨드만 출력
#   CKPT=...      best 외 다른 체크포인트로 `best` 를 돌릴 때
#   PY=...        python 실행기 (기본 ./.venv/bin/python)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
GPU="${GPU:-0}"
CKPT="${CKPT:-finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth}"
cd "$ROOT"

run() {
  echo "+ CUDA_VISIBLE_DEVICES=$GPU ${*/#$PY/python}" >&2
  [ -n "${DRY:-}" ] || CUDA_VISIBLE_DEVICES="$GPU" "$@"
}
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

eval_best() {
  if [ ! -e "$CKPT" ]; then
    echo "체크포인트가 없다: $CKPT" >&2
    echo "  → ./train.sh best status 로 재현 단계를 확인하거나, CKPT=<경로> 로 지정할 것" >&2
    exit 1
  fi
  cat <<'EOF'
best 모델 재현 검증 — 아래 수치가 나와야 한다 (README 「결과 요약」/ docs/EXPERIMENTS.md §5)

                기록    실측 재현(2026-08-14)
  한글 전체     0.127   0.121
  한글 단문     0.286   0.269
  한글 중문     0.049   0.047
  한글 장문     0.059   0.060
  영어 전체     0.045   0.032

  (n=100, cfg=1.0, coherent. 생성은 run-to-run 재현이 안 되므로 ±0.01 수준 차이는 정상이고,
   판정은 n≥100 페어링으로만 한다 — 단발 n=24 수치로 채택/기각하지 말 것.)

EOF
  echo "== 한글 =="
  run "$PY" eval/htr_cer.py --ckpt "$CKPT" --n 100 --coherent "$@"
  echo "== 영어 =="
  run "$PY" eval/htr_cer.py --ckpt "$CKPT" --n 100 --coherent --english "$@"
}

exp_run() {
  if [ -z "${1:-}" ]; then
    echo "experiments/ 스크립트 (실행: ./eval.sh exp <이름> [인자])"; echo
    for f in experiments/*.py; do
      b="$(basename "$f" .py)"
      [ "$b" = "common" ] && continue
      printf "  %-18s %s\n" "$b" "$(sed -n '1s/^"""//p' "$f")"
    done
    return
  fi
  local name="$1"; shift
  [ -f "experiments/$name.py" ] || { echo "experiments/$name.py 가 없다" >&2; exit 1; }
  run "$PY" "experiments/$name.py" "$@"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  best)    eval_best "$@" ;;
  cer)     run "$PY" eval/htr_cer.py "$@" ;;
  easyocr) run "$PY" eval/cer.py "$@" ;;
  echo)    run "$PY" eval/echo_metrics.py "$@" ;;
  all)     run "$PY" eval/all.py "$@" ;;
  fonts)   run "$PY" tools/font_audit.py "$@" ;;
  exp)     exp_run "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "알 수 없는 명령: $cmd" >&2; echo >&2; usage >&2; exit 1 ;;
esac
