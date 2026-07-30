"""pretrained aux 모델(HTR, 옵션 WriterID) 다운로드 헬퍼.

원본 Emuru 의 HTR/WriterID 는 HF 가 아니라 **GitHub Releases** 로 배포된다(HF 는 401).
이 스크립트가 tar.gz 를 받아 `model_zoo/` 에 풀어, train_aux_htr.py / train_vae_korean.py 가
`HTR(**config)+load_state_dict(safetensors)` 로 warm-start 하도록 한다.

원본 HTR 은 라틴 전용(alphabet_size=167). 한글은 train_aux_htr.py 가 vocab 을 확장(warm-start)해 finetune.

사용:
  python tools_fetch_aux.py            # HTR 만
  python tools_fetch_aux.py --wid      # WriterID 도 (현재 파이프라인 미사용)
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZOO = HERE / "model_zoo"

REL = "https://github.com/aimagelab/Emuru-autoregressive-text-img/releases/download"
ASSETS = {
    "htr": (f"{REL}/emuru_vae_htr/emuru_vae_htr.tar.gz", "emuru_vae_htr"),
    "wid": (f"{REL}/emuru_vae_wid/emuru_vae_writer_id.tar.gz", "emuru_vae_writer_id"),
}


def fetch(url: str, dirname: str):
    dst = ZOO / dirname
    if (dst / "config.json").exists() and (dst / "model.safetensors").exists():
        print(f"[skip] {dst} already exists")
        return dst
    ZOO.mkdir(parents=True, exist_ok=True)
    tgz = ZOO / f"{dirname}.tar.gz"
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, tgz)
    print(f"[extract]  {tgz} -> {ZOO}")
    with tarfile.open(tgz) as t:
        t.extractall(ZOO)
    tgz.unlink()
    assert (dst / "config.json").exists(), f"extract 후 {dst}/config.json 없음"
    print(f"[done] {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wid", action="store_true", help="WriterID 도 받기(현재 파이프라인 미사용)")
    args = ap.parse_args()
    fetch(*ASSETS["htr"])
    if args.wid:
        fetch(*ASSETS["wid"])


if __name__ == "__main__":
    main()
