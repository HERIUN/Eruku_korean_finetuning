"""한글 생성 결과를 '자기설명적'으로 그려주는 뷰어.

각 행 = 폰트(스타일) 1개. 열 구성:
  [스타일 ref]            : 조건으로 준 참조 손글씨 (이 스타일로 써라)
  [정답 예시(이 폰트)]    : 목표 텍스트를 이 폰트로 직접 렌더 → "이렇게 나와야 함" 기준
  [생성 cfg=v ...]        : 모델이 실제 생성한 것 (style prefix 잘라 생성분만)

라벨/제목은 NanumGothic 으로 렌더 (한글 표시). 목표 텍스트는 상단 제목에 크게 표시.

Usage:
  python infer_show.py --ckpt finetune_runs/korean_v2/checkpoint_step_072000.pth \
      --seed-text "한국어 OCR 2024" --cfgs 1.0 2.0 3.0 --n-writers 4
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, torchvision, cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eruku_continuous_inf import Emuru

GOTHIC = str(HERE / "assets" / "fonts_label" / "NanumGothic-Regular.ttf")
GOTHIC_B = str(HERE / "assets" / "fonts_label" / "NanumGothic-Bold.ttf")
FONTS_DIR = HERE / "assets" / "fonts_korean_v2" / "train"


def label_img(text, w, h, size, bold=False, center=True, bg=255):
    f = ImageFont.truetype(GOTHIC_B if bold else GOTHIC, size)
    im = Image.new("L", (w, h), bg); d = ImageDraw.Draw(im)
    b = d.textbbox((0, 0), text, font=f); tw, th = b[2] - b[0], b[3] - b[1]
    x = (w - tw) // 2 if center else 5
    d.text((x - b[0], (h - th) // 2 - b[1]), text, font=f, fill=0)
    return np.array(im)


def fit(arr, cw, ch, pad=4):
    """grayscale line 을 cell(cw×ch) 안에 높이맞춰 좌측정렬, 흰 패딩."""
    h, w = arr.shape
    th = ch - 2 * pad
    nw = max(1, int(w * th / h))
    if nw > cw - 2 * pad:
        nw = cw - 2 * pad; th = max(1, int(h * nw / w))
    r = cv2.resize(arr, (nw, th), interpolation=cv2.INTER_AREA)
    canvas = np.full((ch, cw), 255, np.uint8)
    y0 = (ch - th) // 2
    canvas[y0:y0 + th, pad:pad + nw] = r
    return canvas


def cell(content, label, cw, ch, lh=26, highlight=False):
    lab = label_img(label, cw, lh, 15, bold=highlight, bg=210 if highlight else 245)
    block = np.vstack([lab, np.full((1, cw), 0, np.uint8), fit(content, cw, ch)])
    return np.pad(block, ((2, 2), (2, 2)), constant_values=0)


def render_in_font(font_path, text, size=72, pad=14):
    f = ImageFont.truetype(str(font_path), size)
    b = f.getbbox(text); w = max(1, b[2] - b[0]); h = max(1, b[3] - b[1])
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    ImageDraw.Draw(im).text((pad - b[0], pad - b[1]), text, font=f, fill=0)
    return np.array(im)


def load_style(path, h=64):
    img = Image.open(path).convert("RGB"); w, hh = img.size
    img = img.resize((max(1, int(w * (h / hh))), h), Image.BILINEAR)
    return T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])(img)


def gen_arr(model, dec, ref_text, seed_text, cfg, max_new, style_px, device):
    with torch.no_grad():
        img, _ = model.generate(decoder_inputs_embeds_vae=dec, style_text=[ref_text],
                                gen_text=[seed_text], cfg_scale=cfg, max_new_tokens=max_new)
    g = img[:, :, :, style_px:] if img.shape[-1] > style_px else img
    return np.array(torchvision.transforms.ToPILImage()(
        ((g[0] + 1) / 2).clamp(0, 1).cpu()).convert("L"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lines-json", default=str(HERE / "data" / "korean_fontset_v2" / "train_lines.json"))
    ap.add_argument("--out", default=str(HERE / "finetune_runs" / "korean_v2" / "show.png"))
    ap.add_argument("--seed-text", default="한국어 OCR 2024")
    ap.add_argument("--gt-image", action="store_true",
                    help="정답예시=AI Hub 실제 line 이미지, target=해당 텍스트(재구성 평가)")
    ap.add_argument("--cfgs", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    ap.add_argument("--n-writers", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-w", type=int, default=340)
    ap.add_argument("--cell-h", type=int, default=72)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    CW, CH = args.cell_w, args.cell_h

    model = Emuru(t5_checkpoint="google-t5/t5-large",
                  vae_checkpoint="blowing-up-groundhogs/emuru_vae").to(device)
    model.alpha = 1.0
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    st = ck["model"] if "model" in ck else ck
    if any(k.startswith("module.") for k in st):
        st = {k[len("module."):]: v for k, v in st.items()}
    model.load_state_dict(st, strict=False)
    model.drop_text = True
    step = ck.get("step", "?"); del ck, st
    model.eval()

    rows = json.loads(Path(args.lines_json).read_text(encoding="utf-8"))
    by_w = {}
    for r in rows:
        by_w.setdefault(r["person_key"], []).append(r)
    rng = random.Random(args.seed)
    writers = rng.sample(list(by_w), min(args.n_writers, len(by_w)))

    col_labels = ["스타일 ref (조건)", "정답 예시 (이 폰트)"] + [f"생성 cfg={c}" for c in args.cfgs]
    ncol = len(col_labels)
    full_w = ncol * (CW + 4)

    grid = []
    for w in writers:
        ref = rng.choice(by_w[w])
        style_img = load_style(Path(ref["image_path"])).unsqueeze(0).to(device)
        mi = model.get_model_inputs([style_img[0]], None, style_len=style_img.shape[-1], gen_len=None, max_img_len=2048)
        dec = mi["decoder_inputs_embeds"].to(device); spx = dec.shape[1] * 8

        ref_arr = np.array(Image.open(ref["image_path"]).convert("L"))
        if args.gt_image:                       # 정답예시 = AI Hub 실제 이미지, target = 해당 텍스트(재구성)
            target = ref["text"]
            gt_arr = ref_arr.copy()
            gt_label = "정답 예시 (AI Hub 실제)"
        else:
            target = args.seed_text
            fp = FONTS_DIR / f"{w}.ttf"
            gt_arr = render_in_font(fp, target) if fp.exists() else np.full((64, 200), 230, np.uint8)
            gt_label = "정답 예시 (이 폰트)"
        cells = [cell(ref_arr, f"스타일 ref: {w}", CW, CH),
                 cell(gt_arr, gt_label, CW, CH, highlight=True)]
        for c in args.cfgs:
            cells.append(cell(gen_arr(model, dec, ref["text"], target, c, args.max_new_tokens, spx, device),
                              f"생성 cfg={c}", CW, CH))
        row = np.hstack(cells)
        if row.shape[1] < full_w:
            row = np.hstack([row, np.full((row.shape[0], full_w - row.shape[1]), 255, np.uint8)])
        grid.append(row)
        grid.append(np.full((6, row.shape[1]), 80, np.uint8))
        print(f"  {w}: ref={ref['text'][:30]!r}")

    body = np.vstack(grid)
    W = body.shape[1]
    title_txt = ("재구성 평가 (target=각 행의 실제 텍스트, 정답예시=AI Hub 실제 이미지)"
                 if args.gt_image else f'목표 텍스트:  "{args.seed_text}"')
    title = label_img(f'{title_txt}      (step {step}, 각 행 = 스타일 1개)',
                      W, 46, 22, bold=True, center=False, bg=255)
    note = label_img('← 조건(스타일) | 이 폰트로 쓴 정답모양 | 모델이 실제 생성한 것(cfg별) →',
                     W, 28, 14, center=False, bg=255)
    out = np.vstack([title, np.full((2, W), 0, np.uint8), note, np.full((3, W), 0, np.uint8), body])
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
