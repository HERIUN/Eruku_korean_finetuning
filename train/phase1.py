"""Phase 1: 한글 glyph 학습 (논문 pre-training 레시피).

짧은 2~3 어절 pair 로 byte→glyph 매핑을 새로 학습.
설정은 전부 **configs/phase1.yaml** 에 있다 (virtual batch 256, lr 1e-4, 65000 step).

  python train/phase1.py                          # configs/phase1.yaml
  python train/phase1.py --max-steps 20000        # 일부만 CLI 로 덮어쓰기
  python train/phase1.py --config configs/my.yaml # 다른 config
  CUDA_VISIBLE_DEVICES=2 python train/phase1.py   # GPU 지정

산출물: configs/phase1.yaml 의 run.out (기본 finetune_runs/korean_p1)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 저장소 루트

from train.core import parse_args, train

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase1.yaml"


def main():
    train(parse_args(default_config=str(CONFIG)))


if __name__ == "__main__":
    main()
