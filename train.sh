#!/usr/bin/env bash
# 학습 진입점. 모든 학습·데이터 생성 기능을 여기 하나로 부른다.
#
#   ./train.sh best [1-7|all|status]   현재 best 모델 재현 (아래 「재현 파이프라인」)
#   ./train.sh en2ko  [args...]        영어 pretrained → 한글 (권장 시작점)
#   ./train.sh phase1 [args...]        논문 Phase 1 (짧은 어절, glyph 학습)
#   ./train.sh phase2 [args...]        논문 Phase 2 (긴 줄 일반화, phase1 에서 resume)
#   ./train.sh vae    [args...]        VAE 한글 적응 (--htr-checkpoint 필수)
#   ./train.sh htr    [args...]        한글 HTR 리더 (평가·VAE loss 겸용)
#   ./train.sh data   [args...]        학습 데이터 파이프라인 샘플 덤프 (sanity check)
#   ./train.sh refset [args...]        추론용 style ref 세트 생성 (data/ref_set*)
#
# 뒤에 붙인 인자는 그대로 넘어간다:  ./train.sh en2ko --lr 1e-5 --max-steps 5000
# 각 하위 명령의 전체 인자는 `--help` 로:  ./train.sh en2ko --help
#
# 환경변수
#   GPU=2     쓸 GPU (기본 0). CUDA_VISIBLE_DEVICES 로 들어간다
#   DRY=1     실행하지 않고 커맨드만 출력
#   PY=...    python 실행기 (기본 ./.venv/bin/python — 시스템 python 은 의존성이 없다)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
GPU="${GPU:-0}"
cd "$ROOT"

run() {
  echo "+ CUDA_VISIBLE_DEVICES=$GPU ${*/#$PY/python}" >&2
  [ -n "${DRY:-}" ] || CUDA_VISIBLE_DEVICES="$GPU" "$@"
}
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }

# ── best 모델 재현 파이프라인 ────────────────────────────────────────────────
# 각 단계 = (이름, 산출물). 산출물이 다음 단계의 입력이다.
# 실제로 배포본 finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth 를
# 만들어낸 순서 그대로다(train_config.yml 기록 기준). 근거·수치는 docs/EXPERIMENTS.md §4·§5.
BEST_STAGES=(
  "1 base    finetune_runs/korean_en2ko_fixed/checkpoint_step_011000.pth  영어 pretrained → 한글 (20k step, 채택은 s11000)"
  "2 htr     finetune_runs/aux_htr_ko/htr_s20000                          한글 HTR 리더 (VAE loss + 평가 리더)"
  "3 vaedec  finetune_runs/vae_korean_dec/vae_s15000                      VAE decoder-only 워밍업"
  "4 vaefull finetune_runs/vae_korean_full/vae_s15000                     VAE full + HTR 0.3"
  "5 vaemse  finetune_runs/vae_korean_full_mse/vae_s28000                 VAE full pure-L1 ← 릴리즈 VAE"
  "6 readapt finetune_runs/eruku_readapt_fullvae/checkpoint_step_015000.pth  새 VAE latent 로 T5 재적응"
  "7 short   finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth ★단문강조 재적응 = best"
)
# ⚠️ 원본 run 들은 `--val-fonts-dir assets/fonts_korean_unseen` 을 썼는데 그 폴더는 이후 정리됐다.
#    아래 커맨드는 그 인자를 빼서 기본값(--val-font-frac 0.1, 학습 폰트에서 분할)으로 돈다.
#    → val_loss 절대값만 원본과 다르고, 생성·CER 결과에는 영향 없다.
VAL_FONTS=()   # 필요하면 VAL_FONTS=(--val-fonts-dir assets/fonts_korean_v2/test)

best_stage() {
  case "$1" in
    1|base)    run "$PY" train/korean_from_en.py --out finetune_runs/korean_en2ko_fixed \
                 --batch-size 2 --grad-accum 128 --lr 5e-5 --max-img-len 8192 --max-steps 20000 \
                 --num-workers 16 --save-every 1000 --log-every 10 \
                 --val-every 1000 --val-samples 1024 --save-samples 16 "${VAL_FONTS[@]}" "${@:2}" ;;
    2|htr)     run "$PY" train/aux_htr.py --out finetune_runs/aux_htr_ko "${@:2}" ;;
    3|vaedec)  run "$PY" train/vae_korean.py --out finetune_runs/vae_korean_dec \
                 --htr-checkpoint finetune_runs/aux_htr_ko/htr_s20000 \
                 --vae-checkpoint blowing-up-groundhogs/emuru_vae \
                 --train-part decoder --htr-weight 0.3 --kl-weight 1e-6 --noise-prob 0.3 \
                 --batch-size 8 --grad-accum 4 --lr 1e-4 --max-steps 15000 --num-workers 16 \
                 --save-every 2000 --log-every 100 --val-every 500 --val-samples 64 "${@:2}" ;;
    4|vaefull) run "$PY" train/vae_korean.py --out finetune_runs/vae_korean_full \
                 --htr-checkpoint finetune_runs/aux_htr_ko/htr_s20000 \
                 --vae-checkpoint finetune_runs/vae_korean_dec/vae_s15000 \
                 --train-part full --htr-weight 0.3 --kl-weight 1e-6 --noise-prob 0.3 \
                 --batch-size 8 --grad-accum 4 --lr 1e-4 --max-steps 15000 --num-workers 16 \
                 --save-every 2000 --log-every 100 --val-every 500 --val-samples 64 "${@:2}" ;;
    5|vaemse)  run "$PY" train/vae_korean.py --out finetune_runs/vae_korean_full_mse \
                 --htr-checkpoint finetune_runs/aux_htr_ko/htr_s20000 \
                 --vae-checkpoint finetune_runs/vae_korean_full/vae_s15000 \
                 --train-part full --htr-weight 0 --kl-weight 1e-6 --noise-prob 0.3 \
                 --batch-size 8 --grad-accum 4 --lr 1e-4 --max-steps 40000 --num-workers 16 \
                 --save-every 4000 --log-every 200 --val-every 1000 --val-samples 64 "${@:2}" ;;
    6|readapt) run "$PY" train/korean_from_en.py --out finetune_runs/eruku_readapt_fullvae \
                 --resume finetune_runs/korean_en2ko_fixed/checkpoint_step_011000.pth \
                 --vae-checkpoint finetune_runs/vae_korean_full_mse/vae_s28000 --reset-vae \
                 --batch-size 4 --grad-accum 8 --lr 5e-5 --max-img-len 8192 --extra-steps 4000 \
                 --num-workers 16 --save-every 1000 --log-every 20 --val-every 1000 \
                 --save-samples 16 "${VAL_FONTS[@]}" "${@:2}" ;;
    7|short)   run "$PY" train/korean_from_en.py --out finetune_runs/eruku_short_fullmse \
                 --resume finetune_runs/eruku_readapt_fullvae/checkpoint_step_015000.pth \
                 --style-words 1 8 --gen-words 1 12 --max-img-len 4096 --extra-steps 15000 \
                 --batch-size 4 --grad-accum 8 --lr 5e-5 \
                 --num-workers 16 --save-every 2000 --log-every 20 --val-every 2000 \
                 --save-samples 16 "${VAL_FONTS[@]}" "${@:2}" ;;
    *) echo "알 수 없는 단계: $1 (1~7 또는 base/htr/vaedec/vaefull/vaemse/readapt/short)" >&2; exit 1 ;;
  esac
}

best_status() {
  echo "best 모델 재현 파이프라인 — finetune_runs/eruku_short_fullmse/checkpoint_step_030000.pth"
  echo "(근거·수치: docs/EXPERIMENTS.md §4·§5 / 단계 실행: ./train.sh best <번호>)"
  echo
  for s in "${BEST_STAGES[@]}"; do
    read -r n name art desc <<<"$s"
    if [ -e "$art" ]; then mark="✅ 있음"; else mark="⬜ 없음"; fi
    printf "  %s %-8s %s\n      %s  %s\n" "$n" "$name" "$desc" "$mark" "$art"
  done
  echo
  echo "  전부 순서대로:  ./train.sh best all      (GPU 수십 시간. 각 단계는 이어서 resume 가능)"
  echo "  이미 있는 산출물은 다시 만들지 않는다 — 필요한 단계만 번호로 지정할 것."
  echo "  평가:           ./eval.sh best"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  best)
    sel="${1:-status}"; shift || true
    case "$sel" in
      status) best_status ;;
      all)    for s in "${BEST_STAGES[@]}"; do
                read -r n _ art _ <<<"$s"
                if [ -e "$art" ]; then echo "== 단계 $n: 산출물이 이미 있어 건너뜀 ($art)" >&2; continue; fi
                echo "== 단계 $n 시작" >&2; best_stage "$n" "$@"
              done ;;
      *)      best_stage "$sel" "$@" ;;
    esac ;;
  en2ko)  run "$PY" train/korean_from_en.py "$@" ;;
  phase1) run "$PY" train/phase1.py "$@" ;;
  phase2) run "$PY" train/phase2.py "$@" ;;
  vae)    run "$PY" train/vae_korean.py "$@" ;;
  htr)    run "$PY" train/aux_htr.py "$@" ;;
  data)   run "$PY" custom_datasets/korean_split.py "$@" ;;
  refset) run "$PY" custom_datasets/korean_fontset.py "$@" ;;
  ""|-h|--help|help) usage ;;
  *) echo "알 수 없는 명령: $cmd" >&2; echo >&2; usage >&2; exit 1 ;;
esac
