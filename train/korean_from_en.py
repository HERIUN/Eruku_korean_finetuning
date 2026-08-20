"""영어 pretrained → 한글 직접 적응 (Phase 1 생략, 실사용 레시피).

설정은 전부 **configs/korean_from_en.yaml** 에 있다
(virtual batch 256 = 16×16, max_img_len 8192, english_frac 0.15, held-out val).

  python train/korean_from_en.py                          # configs/korean_from_en.yaml
  python train/korean_from_en.py --lr 3e-5                # 일부만 CLI 로 덮어쓰기
  python train/korean_from_en.py --config configs/my.yaml # 다른 config

산출물: configs/korean_from_en.yaml 의 run.out (기본 finetune_runs/korean_en2ko)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 저장소 루트

from train.core import parse_args, train

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "korean_from_en.yaml"


def main():
    train(parse_args(default_config=str(CONFIG)))


if __name__ == "__main__":
    main()
