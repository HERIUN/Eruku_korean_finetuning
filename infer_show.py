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
    # getbbox 가 일부 폰트에서 상/하 여백을 과장 → montage 셀에서 글자가 쭈그러듦.
    # 실제 '잉크 bbox'(그려진 픽셀 min/max)로 타이트하게 crop. 잉크를 전부 포함하므로
    # 글자 부위(받침·디센더)는 절대 안 자름 — 통통 튀는 손글씨 폰트(WagleWagle 등)도 안전.
    # (UlsanJunggu 처럼 '델' 글리프가 본체서 수백px 분리된 malformed 폰트는 이 crop 으로도
    #  거대해지지만, 그 폰트는 eval montage 에서 기본 제외(--exclude-writers)로 처리.)
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    ImageDraw.Draw(im).text((pad - b[0], pad - b[1]), text, font=f, fill=0)
    arr = np.array(im)
    ys, xs = np.where(arr < 250)                       # 잉크(anti-alias 포함) 픽셀
    if len(ys):
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        arr = arr[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    return arr


def load_style(path, h=64):
    img = Image.open(path).convert("RGB"); w, hh = img.size
    img = img.resize((max(1, int(w * (h / hh))), h), Image.BILINEAR)
    return T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])(img)


def gen_arr(model, dec, ref_text, seed_text, cfg, max_new, style_px, device):
    with torch.no_grad():
        img, special = model.generate(decoder_inputs_embeds_vae=dec, style_text=[ref_text],
                                      gen_text=[seed_text], cfg_scale=cfg, max_new_tokens=max_new)
    # 깨끗한 gen 영역만 자르기: 고정 style_px 로 자르면 모델이 마저 그린 style 텍스트 echo 가
    # 남는다. generate 가 돌려주는 special_sequence (3=style img, 2=image, 0=SOG, 1=EOG)
    # 에서 SOG 위치가 style(+echo)→gen 의 진짜 경계 (continue_gen_test 참고).
    sp = special.repeat_interleave(8)[:img.shape[-1]]   # latent→pixel (8px/latent)
    sog = (sp == 0).nonzero().flatten()
    if len(sog) > 0:
        g = img[:, :, :, int(sog[0].item()) + 1:]       # 첫 SOG 이후 = gen 텍스트
    else:
        g = img[:, :, :, style_px:] if img.shape[-1] > style_px else img
    return np.array(torchvision.transforms.ToPILImage()(
        ((g[0] + 1) / 2).clamp(0, 1).cpu()).convert("L"))


def load_model(ckpt, device, vae_checkpoint=None):
    """Emuru 모델 로드 + 체크포인트 적용. (infer_show/infer_matrix 공용)

    vae_checkpoint 를 주면 그 VAE 로 모델을 만들고, **ckpt 의 옛 vae.\\* 키를 strip** 해서
    새 VAE 가 덮이지 않게 한다(한글 적응 VAE 교체 검증용; ckpt.model 에 옛 vae.* 252키가 있음).
    안 주면 기존 동작(default emuru_vae, ckpt 의 vae.* 그대로 로드).
    """
    vae_ck = vae_checkpoint or "blowing-up-groundhogs/emuru_vae"
    model = Emuru(t5_checkpoint="google-t5/t5-large", vae_checkpoint=vae_ck).to(device)
    model.alpha = 1.0
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    st = ck["model"] if "model" in ck else ck
    if any(k.startswith("module.") for k in st):
        st = {k[len("module."):]: v for k, v in st.items()}
    if vae_checkpoint:  # 새 VAE 유지: ckpt 의 옛 vae.* 제거
        n_before = len(st)
        st = {k: v for k, v in st.items() if not k.startswith("vae.")}
        print(f"[load_model] vae swapped -> {vae_ck} (ckpt vae.* {n_before - len(st)}키 strip)")
    model.load_state_dict(st, strict=False)
    model.drop_text = True
    step = ck.get("step", "?"); del ck, st
    model.eval()
    return model, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae-checkpoint", default=None,
                    help="한글 적응 VAE(vae_sXXXX) 로 교체해 생성. 주면 ckpt 의 옛 vae.* 를 strip")
    ap.add_argument("--lines-json", default=str(HERE / "data" / "korean_fontset_v2" / "train_lines.json"))
    ap.add_argument("--out", default=str(HERE / "finetune_runs" / "korean_v2" / "show.png"))
    ap.add_argument("--seed-text", default="한국어 OCR 2024")
    ap.add_argument("--row-texts", nargs="+", default=None,
                    help="행마다 다른 목표 텍스트. 주면 writer 행 i 는 row_texts[i %% len] 사용 "
                         "(폰트+텍스트 동시 다양화). 안 주면 모든 행에 --seed-text 적용")
    ap.add_argument("--echo-style", action="store_true",
                    help="각 행의 목표 텍스트를 그 행 style ref 의 텍스트와 동일하게 설정. "
                         "style_text == target → 모델이 '본 글자'를 그대로 재현하는지 확인용. "
                         "(--row-texts/--seed-text 보다 우선)")
    ap.add_argument("--cfgs", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument("--n-writers", type=int, default=4,
                    help="montage 에 넣을 writer(폰트) 수 = 행 개수. lines-json 의 person_key "
                         "중에서 --seed 로 무작위 추첨. 폰트 총수보다 크면 전체로 clamp. "
                         "클수록 다양한 필체를 보지만 생성 횟수(writer×cfg)가 늘어 느려짐")
    ap.add_argument("--max-new-tokens", type=int, default=220,
                    help="자기회귀 생성 토큰(8px latent slice) 상한. 1 토큰 ≈ 가로 8px. "
                         "모델이 EOG 를 내면 그 전에 멈추고, 안 내면 이 값에서 강제 종료. "
                         "너무 작으면 긴 문장이 중간에 잘리고, 너무 크면 느려짐 "
                         "(기본 220 ≈ 가로 1760px 까지 생성 가능)")
    ap.add_argument("--seed", type=int, default=0,
                    help="writer 추첨 + 각 행에서 어떤 ref 라인을 고를지 결정하는 난수 시드. "
                         "같은 시드 = 같은 폰트·같은 ref 선택(체크포인트 간 공정 비교). "
                         "시드를 바꾸면 다른 폰트 조합을 본다")
    ap.add_argument("--cell-w", type=int, default=340,
                    help="montage 각 칸(셀)의 가로 픽셀. 생성/정답 이미지를 이 폭에 맞춰 리사이즈. "
                         "키우면 글자가 크게 보이지만 전체 montage 파일도 넓어짐(시각화 전용, 품질 무관)")
    ap.add_argument("--cell-h", type=int, default=72,
                    help="montage 각 칸(셀)의 세로 픽셀. 한 줄 라인 이미지 높이에 맞춤 "
                         "(시각화 전용, 모델 입출력 해상도와 무관)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-style-text", action="store_true",
                    help="style_text(ref 전사)를 주지 않고 빈 문자열로 생성 (style 이미지만으로 "
                         "스타일 파악). 줬을 때와 비교용. 컬럼 라벨에 (no-stext) 표시")
    ap.add_argument("--exclude-writers", nargs="*", default=["UlsanJunggu"],
                    help="montage writer(폰트) 후보에서 제외할 key. 기본: 깨진 글리프('델') 폰트 "
                         "UlsanJunggu (font_audit 참고). '#N' 접미사는 자동 무시하고 base 이름으로 비교.")
    args = ap.parse_args()
    device = torch.device(args.device)
    CW, CH = args.cell_w, args.cell_h

    model, step = load_model(args.ckpt, device, vae_checkpoint=args.vae_checkpoint)

    rows = json.loads(Path(args.lines_json).read_text(encoding="utf-8"))
    by_w = {}
    for r in rows:
        by_w.setdefault(r["person_key"], []).append(r)
    # 제외 폰트(예: 깨진 글리프 UlsanJunggu) 는 writer 후보에서 뺌 ('#N' 접미사 무시)
    if args.exclude_writers:
        excl = set(args.exclude_writers)
        by_w = {w: v for w, v in by_w.items() if w.split("#")[0] not in excl}
    rng = random.Random(args.seed)
    writers = rng.sample(list(by_w), min(args.n_writers, len(by_w)))

    stext_tag = " (no-stext)" if args.no_style_text else ""
    col_labels = ["스타일 ref (조건)", "정답 예시 (이 폰트)"] + [f"생성 cfg={c}{stext_tag}" for c in args.cfgs]
    ncol = len(col_labels)
    full_w = ncol * (CW + 4)

    grid = []
    for i, w in enumerate(writers):
        ref = rng.choice(by_w[w])
        style_img = load_style(Path(ref["image_path"])).unsqueeze(0).to(device)
        mi = model.get_model_inputs([style_img[0]], None, style_len=style_img.shape[-1], gen_len=None, max_img_len=2048)
        dec = mi["decoder_inputs_embeds"].to(device); spx = dec.shape[1] * 8

        ref_arr = np.array(Image.open(ref["image_path"]).convert("L"))
        # 정답 예시 = 목표 텍스트를 해당 폰트로 렌더 (모델이 이 폰트 스타일로 그려야 할 목표 모양)
        # --echo-style 면 target=ref 텍스트(본 글자 재현), --row-texts 면 행마다 다름, 아니면 --seed-text.
        if args.echo_style:
            target = ref["text"]
        elif args.row_texts:
            target = args.row_texts[i % len(args.row_texts)]
        else:
            target = args.seed_text
        # 폰트 파일 찾기: ① row 에 font_path 있으면 그걸 (unseen/외부 폰트 지원),
        # ② 없으면 FONTS_DIR/{base}.ttf. person_key 의 '#N'(같은 폰트 여러 ref 구분용) 접미사 제거.
        base = w.split("#")[0]
        fp = Path(ref["font_path"]) if ref.get("font_path") else FONTS_DIR / f"{base}.ttf"
        gt_arr = render_in_font(fp, target) if fp.exists() else np.full((64, 200), 230, np.uint8)
        gt_label = f"정답 폰트: {base}"
        # 행마다 텍스트가 다르면 정답칸 라벨에 목표 텍스트를 같이 보여줌
        if args.echo_style or args.row_texts:
            gt_label = f"목표(=ref): {target[:20]}" if args.echo_style else f"목표: {target[:22]}"
        cells = [cell(ref_arr, f"스타일 ref: {w}", CW, CH),
                 cell(gt_arr, gt_label, CW, CH, highlight=True)]
        style_text_in = "" if args.no_style_text else ref["text"]
        for c in args.cfgs:
            cells.append(cell(gen_arr(model, dec, style_text_in, target, c, args.max_new_tokens, spx, device),
                              f"생성 cfg={c}{stext_tag}", CW, CH))
        row = np.hstack(cells)
        if row.shape[1] < full_w:
            row = np.hstack([row, np.full((row.shape[0], full_w - row.shape[1]), 255, np.uint8)])
        grid.append(row)
        grid.append(np.full((6, row.shape[1]), 80, np.uint8))
        print(f"  {w}: ref={ref['text'][:30]!r}")

    body = np.vstack(grid)
    W = body.shape[1]
    if args.echo_style:
        title_txt = '목표 텍스트 = style ref 텍스트 (본 글자 그대로 재현되는지)'
    elif args.row_texts:
        title_txt = '목표 텍스트:  행마다 다름 (각 행 정답칸 라벨 참고)'
    else:
        title_txt = f'목표 텍스트:  "{args.seed_text}"'
    title = label_img(f'{title_txt}      (step {step}, 각 행 = 스타일 1개)',
                      W, 46, 22, bold=True, center=False, bg=255)
    note = label_img('← 조건(스타일) | 이 폰트로 쓴 정답모양 | 모델이 실제 생성한 것(cfg별) →',
                     W, 28, 14, center=False, bg=255)
    out = np.vstack([title, np.full((2, W), 0, np.uint8), note, np.full((3, W), 0, np.uint8), body])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"saved {args.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
