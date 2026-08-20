"""Phase 2: 긴 줄 적응 (논문 5000 step 추가).

Phase 1 ckpt 에서 resume 해 긴 style(1~8)/gen(1~32) 으로 이어 학습.
설정은 전부 **configs/phase2.yaml** 에 있다 (virtual batch 256, extra 5000 step, p_drop 0.10).

  python train/phase2.py                          # configs/phase2.yaml
  python train/phase2.py --extra-steps 3000        # 일부만 CLI 로 덮어쓰기
  python train/phase2.py --config configs/my.yaml # 다른 config
  CUDA_VISIBLE_DEVICES=2 python train/phase2.py   # GPU 지정

산출물: configs/phase2.yaml 의 run.out (기본 finetune_runs/korean_p2)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 저장소 루트

from train.core import parse_args, train

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "phase2.yaml"


def main():
    train(parse_args(default_config=str(CONFIG)))


if __name__ == "__main__":
    main()
