"""Phase 2: 긴 줄 일반화 (논문 fine-tuning 레시피).

Phase 1 ckpt 에서 resume, style 1~8 / gen 1~32 어절로 길이 일반화만 학습.
virtual batch 256 = batch 2 × grad-accum 128,  lr 1e-4,  +5000 step.

  python train/phase2.py                 # Phase1 의 checkpoint_last.pth 에서 resume
  CUDA_VISIBLE_DEVICES=2 python train/phase2.py
  python train/phase2.py --resume finetune_runs/korean_p1/checkpoint_step_065000.pth

핵심: --extra-steps 5000 → max_steps = resume_step(65000) + 5000 = 70000 자동계산.
      (max-steps 를 5000 으로 주면 step 카운터가 65000 에서 시작해 즉시 종료하는 사고를 방지)

산출물: finetune_runs/korean_p2/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 저장소 루트

from train.core import make_parser, train


def main():
    p = make_parser()
    p.set_defaults(
        online_split=True,
        style_words=[1, 8], gen_words=[1, 32],
        text_dropout=0.05, style_text_dropout=0.10,
        batch_size=2, grad_accum=128, lr=1e-4,
        num_workers=16,
        extra_steps=5000,                 # resume step + 5000 = 누적 max_steps 자동
        save_every=250, log_every=10,
        save_samples=16,
        resume="finetune_runs/korean_p1/checkpoint_last.pth",
        out="finetune_runs/korean_p2",
    )
    train(p.parse_args())


if __name__ == "__main__":
    main()
