"""공식 Eruku pretrained 체크포인트 다운로드 (HuggingFace).

- model_zoo/eruku_pretrained/000073688.pth  (~8.5 GB, 영어 pretrained full ckpt)
- files/checkpoints/Origami_bw_img/origami.pth  (~42 MB, OrigamiNet OCR — 모델 init 에 필요)

Usage:
  uv run python scripts/download_pretrained.py
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "blowing-up-groundhogs/eruku"
ROOT = Path(__file__).resolve().parent.parent


def fetch(filename: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    out = local_dir / filename
    if out.exists():
        print(f"skip (exists): {out}")
        return
    print(f"downloading {REPO}/{filename} -> {out}")
    hf_hub_download(repo_id=REPO, filename=filename, local_dir=str(local_dir))
    print(f"done: {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    fetch("origami.pth", ROOT / "files" / "checkpoints" / "Origami_bw_img")
    fetch("000073688.pth", ROOT / "model_zoo" / "eruku_pretrained")
    print("\nall checkpoints ready.")
