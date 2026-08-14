"""Phase 1: 한글 glyph 학습 (논문 pre-training 레시피).

짧은 2~3 어절 pair 로 byte→glyph 매핑을 새로 학습.
virtual batch 256 = batch 128 × grad-accum 2,  lr 1e-4,  65000 step.

  python train/phase1.py                 # 기본 (GPU 는 CUDA_VISIBLE_DEVICES 로 지정)
  CUDA_VISIBLE_DEVICES=2 python train/phase1.py
  python train/phase1.py --max-steps 20000   # 개별 인자 덮어쓰기 가능

산출물: finetune_runs/korean_p1/  (정상 완료 시 checkpoint_last.pth → Phase 2 가 resume)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 저장소 루트

from train.core import make_parser, train


def main():
    p = make_parser()
    p.set_defaults(
        online_split=True,
        style_words=[2, 3], gen_words=[2, 3],
        text_dropout=0.05,
        batch_size=128, grad_accum=2, lr=1e-4,
        num_workers=16,
        max_steps=65000, save_every=2500, log_every=10,
        save_samples=16,
        out="finetune_runs/korean_p1",
    )
    train(p.parse_args())


if __name__ == "__main__":
    main()
