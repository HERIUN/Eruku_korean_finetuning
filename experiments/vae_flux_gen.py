"""§13 2단계: 같은 style·텍스트·시드로 두 ckpt 생성 비교 몽타주 (행 = style | A | B).

--set ko     : 실제 스캔(custom_datasets/hw_line_sample, ink bbox 크롭, 전사 줌) + 학습 폰트 2종, 한글 텍스트
      ko_test: 미노출 한글 폰트(fonts_korean_v2/test) 4종, 한글 텍스트
      en     : 영어 폰트 4종, 영어 텍스트
      aihw   : AI 손글씨 폰트 4종(assets/custom_hw_fonts, 평가 전용), 한글 텍스트
      scan   : 실제 스캔 6장(ko 세트의 3장 제외), 한글 텍스트
      iam    : IAM test 실제 손글씨 라인 4명(hwd IAMLines, 전사 줌), 영어 텍스트
  .venv/bin/python experiments/vae_flux_gen.py --a finetune_runs/eruku_corpusmax/checkpoint_last.pth \
      --b finetune_runs/eruku_flux1vae/checkpoint_step_002000.pth --out docs/img_vae/flux_gen_s2000.png
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R))
from infer.show import load_model, style_tensor, gen_from_style, render_in_font
import tools.hf_upload as hu

FONT = ImageFont.truetype(str(R / "assets/fonts_korean_v2/train/GothicA1.ttf"), 15)
TEXTS = {"ko": ["한국어 손글씨 생성", "밝은 햇살 붉은 노을이 강을 물들였다", "형형색색 꽃들이 활짝 피어난 봄날의 공원"],
         "en": ["Handwriting generation", "The quick brown fox jumps over the lazy dog", "Bright sunlight painted the river red at dusk"]}


def font_styles(d, names, t):
    return [(f"폰트 {f}", render_in_font(R / d / f"{f}.ttf", t), t) for f in names]


def iam_styles(n=4):
    from hwd.datasets.shtg import IAMLines
    ds = IAMLines(num_style_samples=1)
    out = []
    for i in range(0, len(ds), len(ds) // n)[:n]:   # 필기자 고르게 등간격
        it = ds[i]
        out.append((f"IAM {i}", np.array(it["style_imgs"][0].convert("L")), it["style_imgs_text"][0]))
    return out


def scan_styles(names):
    ns = {"np": np, "Image": Image}; exec(hu.BBOX_HELPER, ns)
    src = R / "custom_datasets/hw_line_sample"
    lab = dict(l.split("\t", 1) for l in (src / "label.txt").read_text().splitlines() if "\t" in l)
    return [(f"스캔 {k[5:8]}", np.array(ns["ink_bbox_crop"](Image.open(src / k)).convert("L")), lab[k]) for k in names]


def styles(kind):
    if kind == "aihw":
        return [(f"AI {f.name.split('-')[1]}", render_in_font(f, "조용히 책을 읽었다"), "조용히 책을 읽었다")
                for f in sorted((R / "assets/custom_hw_fonts").glob("*.otf"))]
    if kind == "scan":
        return scan_styles([f"line_{i:03d}.png" for i in (2, 4, 8, 10, 15, 18)])
    if kind == "ko_test":
        return font_styles("assets/fonts_korean_v2/test", ("NanumPenScript", "Gaegu", "HiMelody", "Dokdo"), "조용히 책을 읽었다")
    if kind == "en":
        return font_styles("assets/fonts_english", ("Caveat", "IndieFlower", "Kalam", "DancingScript"), "I read a book quietly")
    if kind == "iam":
        return iam_styles()
    ns = {"np": np, "Image": Image}; exec(hu.BBOX_HELPER, ns)
    src = R / "custom_datasets/hw_line_sample"
    lab = dict(l.split("\t", 1) for l in (src / "label.txt").read_text().splitlines() if "\t" in l)
    out = [(f"스캔 {k}", np.array(ns["ink_bbox_crop"](Image.open(src / k)).convert("L")), lab[k])
           for k in ("line_000.png", "line_005.png", "line_012.png")]
    for f in ("GamjaFlower", "Diary"):
        fp = next((R / "assets/fonts_korean_v2/train").glob(f"{f}*.ttf"))
        t = "조용히 책을 읽었다"; out.append((f"폰트 {f}", render_in_font(fp, t), t))
    return out


def main(a):
    dev = torch.device("cuda"); st = styles(a.set); texts = TEXTS["en" if a.set in ("en", "iam") else "ko"]; res = {}
    for name in ("a", "b"):
        m, step = load_model(getattr(a, name), dev)
        res[name] = (step, [[gen_from_style(m, style_tensor(arr).to(dev), stext, t, a.cfg, 480, dev, seed=0)
                             for t in texts] for _, arr, stext in st])
        del m; torch.cuda.empty_cache()
    rows = []
    for i, (lab, arr, _) in enumerate(st):
        sty = Image.fromarray(arr); sty = sty.resize((max(1, round(sty.width * 64 / sty.height)), 64))
        rows.append((f"style: {lab}", sty))
        for j in range(len(texts)):
            for name in ("a", "b"):   # 라벨 = run 폴더명(eruku_ 뗌) + step
                run = Path(getattr(a, name)).parent.name.removeprefix("eruku_")
                rows.append((f"{run} s{res[name][0]}", Image.fromarray(res[name][1][i][j])))
        rows.append(None)
    W = min(1800, max(r[1].width for r in rows if r) + 150); H = sum(68 if r else 10 for r in rows)
    c = Image.new("L", (W, H), 255); d = ImageDraw.Draw(c); y = 0
    for r in rows:
        if r is None:
            d.line([(0, y + 5), (W, y + 5)], fill=140); y += 10; continue
        d.text((6, y + 24), r[0], font=FONT, fill=0); c.paste(r[1], (145, y + 2)); y += 68
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); c.save(a.out); print("saved", a.out, c.size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--cfg", type=float, default=1.0); ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=["ko", "ko_test", "aihw", "scan", "en", "iam"], default="ko")
    main(ap.parse_args())
