"""(§13) VAE 왕복 충실도: 입력(높이 64 bilinear, vae_recon.py 와 같은 리사이즈) vs 복원.
MSE·SSIM(전체), 잉크 영역 MAE, 회색비율(잉크 픽셀 중 50<v<200 비율) 입력 vs 복원, 순흑(<30) 비율."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
S = Path(sys.argv[1])
for s in ("font", "scan"):
    tr = json.loads((S / f"src_{s}/transcriptions.json").read_text())
    ref = {}
    for k in tr:
        im = Image.open(S / f"src_{s}" / k).convert("L"); w, h = im.size
        ref[k] = np.asarray(im.resize((max(1, round(w * 64 / h)), 64), Image.BILINEAR), np.float32)
    (S / f"ref_{s}").mkdir(exist_ok=True)
    for k, a in ref.items(): Image.fromarray(a.astype(np.uint8)).save(S / f"ref_{s}" / k)
    (S / f"ref_{s}/transcriptions.json").write_text(json.dumps(tr, ensure_ascii=False))
    def stats(get):
        m, ss, ink, gray, black = [], [], [], [], []
        for k, a in ref.items():
            b = get(k); mask = a < 200
            m.append(((a - b) / 255) ** 2 .mean() if False else (((a - b) / 255) ** 2).mean())
            ss.append(ssim(a, b, data_range=255))
            ink.append(np.abs(a - b)[mask].mean() / 255 if mask.any() else 0)
            bm = b < 200
            gray.append(((b > 50) & (b < 200))[bm].mean() if bm.any() else 0)
            black.append((b < 30)[bm].mean() if bm.any() else 0)
        return np.mean(m), np.mean(ss), np.mean(ink), np.mean(gray), np.mean(black)
    print(f"[{s}] n={len(ref)}   MSE↓   SSIM↑   잉크MAE↓  회색비율  순흑비율")
    r = stats(lambda k: ref[k]); print(f"  input        —       —       —        {r[3]:.3f}    {r[4]:.3f}")
    for v in ("eruku", "flux1", "flux2"):
        r = stats(lambda k: np.asarray(Image.open(S / f"{v}_{s}" / k).convert("L"), np.float32))
        print(f"  {v:7s}  {r[0]:.4f}  {r[1]:.3f}   {r[2]:.3f}    {r[3]:.3f}    {r[4]:.3f}")
