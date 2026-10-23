"""decode_ab 의 all/gen/ctx 를 같은 샘플끼리 나란히 (모델별 1장)."""
import json, sys, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
D = Path("/data/work/dgkang/Eruku_korean_finetuning/finetune_runs/_iam/decode_ab"); out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "img_decode"
H, W, LW = 48, 1100, 150
def load(p):
    im = Image.open(p).convert("L"); im = im.resize((max(1, im.width * H // im.height), H))
    return im.crop((0, 0, min(im.width, W - LW), H))
def dark(p): return (np.array(Image.open(p).convert("L")) < 128).mean()
for m in ("pretrained", "corpusmax"):
    keys = sorted(json.load(open(D / f"{m}_noTs/all/fake/transcriptions.json")))
    # all vs ctx(w/o T_s) 잉크량 차이 큰 것 3개 + 등간격 3개
    diff = sorted(keys, key=lambda k: -abs(dark(D/f"{m}_noTs/all/fake"/k) - dark(D/f"{m}_noTs/ctx/fake"/k)))
    pick = diff[:3] + [k for k in keys[::len(keys)//3][:3] if k not in diff[:3]]
    rows = []
    for k in pick:
        txt = json.load(open(D / f"{m}_Ts/all/fake/transcriptions.json"))[k]
        rows.append((f"{k}: {txt[:110]}", None))
        rows.append(("real", D / f"{m}_Ts/all/real" / k))
        for t in ("Ts", "noTs"):
            for d in ("all", "gen", "ctx"):
                rows.append((f"{'w/' if t=='Ts' else 'w/o'} T_s {d}", D / f"{m}_{t}/{d}/fake" / k))
        rows.append(("", "gap"))
    canvas = Image.new("L", (W, sum(18 if p is None else 10 if p == "gap" else H + 4 for _, p in rows)), 255)
    dr, y = ImageDraw.Draw(canvas), 0
    for lab, p in rows:
        if p is None: dr.text((4, y + 3), lab, fill=0); y += 18; continue
        if p == "gap": dr.line((0, y + 5, W, y + 5), fill=160); y += 10; continue
        dr.text((4, y + H // 2 - 6), lab, fill=0 if lab != "real" else 90)
        canvas.paste(load(p), (LW, y + 2)); y += H + 4
    canvas.save(out / f"decode_{m}.png"); print(out / f"decode_{m}.png", canvas.size)
