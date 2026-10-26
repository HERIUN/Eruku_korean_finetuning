"""§13 VAE 왕복 비교 몽타주: 입력 / 우리 VAE / FLUX.1 VAE / Flux2 VAE (finetune_runs/_vaecmp)."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R = Path(__file__).resolve().parents[1]; D = R / "finetune_runs/_vaecmp"
FONT = ImageFont.truetype(str(R / "assets/fonts_korean_v2/train/GothicA1.ttf"), 15)
ks = list(json.loads((D / "src_font/transcriptions.json").read_text()))
rows = []
for s, k in [("scan", "line_000.png"), ("scan", "line_005.png"), ("scan", "line_012.png"), ("font", ks[3]), ("font", ks[40])]:
    for v, lab in [("ref", "입력"), ("eruku", "우리 VAE"), ("flux1", "FLUX.1 VAE"), ("flux2", "Flux2 VAE")]:
        rows.append((lab, Image.open(D / f"{v}_{s}" / k).convert("L")))
    rows.append(None)
W = max(r[1].size[0] for r in rows if r) + 110; H = sum(68 if r else 10 for r in rows)
c = Image.new("L", (W, H), 255); d = ImageDraw.Draw(c); y = 0
for r in rows:
    if r is None:
        d.line([(0, y + 5), (W, y + 5)], fill=140); y += 10; continue
    d.text((6, y + 24), r[0], font=FONT, fill=0); c.paste(r[1], (105, y + 2)); y += 68
c.save(R / "docs/img_vae/vae_flux_cmp.png")
