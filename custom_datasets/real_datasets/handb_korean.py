"""Eruku 용 hanDB Korean dataset adapter.

`Base_dataset` 를 상속하지 않고, 같은 인터페이스 (sample dict with style_/same_/other_) 만 맞춤.
이미지는 hanDB 의 PNG 원본 — runtime 에서 grayscale 64-height resize.

ByT5 tokenizer 가 한글 byte 처리하므로 charset 정의는 별도 필요 없음.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class HanDBKoreanDataset(Dataset):
    def __init__(self, lines_json: str, img_height: int = 64, batch_keys=None):
        self.img_height = img_height
        self.batch_keys = batch_keys if batch_keys is not None else ["style", "other", "same"]

        rows = json.loads(Path(lines_json).read_text(encoding="utf-8"))
        self.imgs: list[Path] = []
        self.imgs_to_label: dict[str, str] = {}
        self.imgs_to_author: dict[str, str] = {}
        author_to_imgs: dict[str, list[str]] = defaultdict(list)

        for r in rows:
            p = Path(r["image_path"])
            if not p.exists():
                continue
            stem = p.stem
            self.imgs.append(p)
            self.imgs_to_label[stem] = r["text"]
            self.imgs_to_author[stem] = r["person_key"]
            author_to_imgs[r["person_key"]].append(stem)

        self.author_to_imgs: dict[str, set[str]] = {
            k: set(v) for k, v in author_to_imgs.items()
        }
        self.imgs_set: set[str] = set(self.imgs_to_label.keys())
        self.stem_to_idx: dict[str, int] = {p.stem: i for i, p in enumerate(self.imgs)}
        # emuru_vae 는 in=3ch(RGB) / out=1ch(grayscale) 비대칭 VAE.
        # encode 입력은 반드시 3채널이어야 하므로 RGB 로 로드 (grayscale PNG → 3ch 복제).
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        print(f"HanDBKoreanDataset: {len(self.imgs)} imgs, {len(self.author_to_imgs)} authors")

    def __len__(self) -> int:
        return len(self.imgs)

    def _load(self, stem: str):
        # find Path by stem
        idx = self.stem_to_idx[stem]
        p = self.imgs[idx]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        new_w = max(1, int(w * (self.img_height / h)))
        img = img.resize((new_w, self.img_height), Image.BILINEAR)
        tensor = self.transform(img)  # [1, H, W] in [-1, 1]
        text = self.imgs_to_label[stem]
        author = self.imgs_to_author[stem]
        return tensor, tensor.shape[-1], text, author

    def _fill(self, sample: dict, stem: str, tag: str):
        img, img_len, text, author = self._load(stem)
        sample[f"{tag}_img"] = img
        sample[f"{tag}_img_len"] = img_len
        sample[f"{tag}_text"] = text
        sample[f"{tag}_author"] = author

    def __getitem__(self, idx: int) -> dict:
        sample: dict = {}
        style_stem = self.imgs[idx].stem
        self._fill(sample, style_stem, "style")
        author = sample["style_author"]

        if "same" in self.batch_keys:
            same_imgs = self.author_to_imgs[author]
            same_stem = random.choice(list(same_imgs))
            self._fill(sample, same_stem, "same")
        if "other" in self.batch_keys:
            other_imgs = self.imgs_set - self.author_to_imgs[author]
            other_imgs = other_imgs if other_imgs else self.author_to_imgs[author]
            other_stem = random.choice(list(other_imgs))
            self._fill(sample, other_stem, "other")
        return sample


def handb_collate(batch: list[dict]) -> dict:
    """Eruku 의 HFDataCollector 비슷한 collate — 가변 길이 image 라 list 로만 모음."""
    out: dict = {}
    for k in batch[0]:
        out[k] = [b[k] for b in batch]
    return out
