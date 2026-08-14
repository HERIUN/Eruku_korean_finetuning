"""HTR 기반 생성 CER 평가 — easyocr 대신 우리 한글 HTR(CER~0.02, 렌더바닥 0.00)을 리더로.

문제: eval/cer.py 의 easyocr 은 한글을 못 읽어 GT 렌더조차 CER 0.236 → 생성 개선을 측정 못 함
      (한글 gen CER 0.20 이 이미 OCR 바닥 아래라 saturated).
해결: train.aux_htr 로 만든 한글 HTR 은 렌더를 CER 0.00 로 읽음 → 바닥이 0 이라 실제 생성오차를 정밀 측정.

생성(reuse infer.show.gen_arr) → HTR greedy 읽기 → target 과 CER. GT 렌더도 HTR 로 읽어 '바닥' 동시 보고.
길이버킷·한/영 분해. (독립 교차검증은 easyocr 쓰는 eval/cer.py 병행.)

사용: CUDA_VISIBLE_DEVICES=2 python eval/htr_cer.py --ckpt <ckpt> [--vae-checkpoint <vae>] [--english] --n 100
"""
from __future__ import annotations
import argparse, random, re, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))
from infer.show import load_model, render_in_font, style_tensor, gen_from_style
from eval.echo_metrics import build_texts, font_can_render
from train.aux_htr import load_pretrained_htr, cer as char_cer
from custom_datasets.korean_alphabet import get_korean_alphabet
from custom_datasets.subsequent_mask import subsequent_mask
from custom_datasets.korean_aux import _fit_width

ALPHA = get_korean_alphabet()


def to_htr_input(arr, dev):
    """grayscale uint8 [H,W](ink~0 bg~255) → [1,1,64,768] [-1,1] (HTR 입력)."""
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)[None, None]
    w64 = max(1, int(round(64 * arr.shape[1] / arr.shape[0])))
    t = F.interpolate(t, size=(64, w64), mode="bilinear", align_corners=False).clamp(0, 1)
    return _fit_width((t * 2 - 1)[0]).unsqueeze(0).to(dev)


@torch.no_grad()
def htr_read(htr, imgs, dev, max_len=150):
    """[B,1,64,768] → 텍스트 리스트 (greedy AR 디코딩)."""
    B = imgs.size(0)
    tok = torch.full((B, 1), ALPHA.sos, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(max_len):
        L = tok.size(1)
        tm = subsequent_mask(L).to(dev)
        pad = torch.zeros(B, L, dtype=torch.bool, device=dev)
        out = htr(imgs, tok, tm, pad)
        nxt = out[:, -1].argmax(-1, keepdim=True)
        tok = torch.cat([tok, nxt], 1)
        done |= (nxt.squeeze(1) == ALPHA.eos)
        if done.all():
            break
    return ALPHA.decode(tok[:, 1:], [ALPHA.eos])


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae-checkpoint", default=None)
    ap.add_argument("--htr-checkpoint", default="finetune_runs/aux_htr_ko/htr_s20000")
    ap.add_argument("--fonts-dir", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--max-words", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--english", action="store_true")
    ap.add_argument("--coherent", action="store_true", help="랜덤단어 대신 실제 문장(권장)")
    ap.add_argument("--no-style-text", action="store_true")
    ap.add_argument("--out", default=str(HERE / "finetune_runs" / "htr_cer.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = random.Random(args.seed)

    model, step = load_model(args.ckpt, dev, vae_checkpoint=args.vae_checkpoint)
    htr = load_pretrained_htr(Path(args.htr_checkpoint)).to(dev).eval()
    assert htr.fc.out_features == len(ALPHA), "HTR alphabet 불일치"
    fonts = sorted(p for p in Path(args.fonts_dir).glob("*.ttf"))
    texts = build_texts(args.n, rng, english=args.english, coherent=args.coherent, max_words=args.max_words)

    per = []  # (nw, cer_gen, cer_gt)
    done = 0
    for text in texts:
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text)), None)
        if fp is None:
            continue
        gt = render_in_font(fp, text)
        stext = "" if args.no_style_text else text
        try:
            gen = gen_from_style(model, style_tensor(gt), stext, text,
                                 args.cfg, args.max_new_tokens, dev)
        except Exception as e:
            print(f"  [skip] {e}"); continue
        hyp_gen = htr_read(htr, to_htr_input(gen, dev), dev)[0]
        hyp_gt = htr_read(htr, to_htr_input(gt, dev), dev)[0]
        cg = char_cer([norm(hyp_gen)], [norm(text)])
        cgt = char_cer([norm(hyp_gt)], [norm(text)])
        per.append((len(text.split()), cg, cgt)); done += 1
        if done % 20 == 0:
            print(f"  {done}/{args.n} ... (누적 gen CER {np.mean([p[1] for p in per]):.3f})")

    def agg(rows):
        return np.mean([p[1] for p in rows]), np.mean([p[2] for p in rows])
    lines = [f"=== HTR-reader CER 평가 (step {step}, n={done}, cfg={args.cfg}, "
             f"{'영어' if args.english else '한글'}{', no-stext' if args.no_style_text else ''}) ===",
             f"reader: 한글 HTR {args.htr_checkpoint} (렌더바닥 ~0.00) | ckpt vae: {args.vae_checkpoint or 'ckpt내장'}",
             f"{'':14s} {'gen CER↓':>9} {'GT(HTR바닥)':>11} {'gap':>7}"]
    cg, cgt = agg(per)
    lines.append(f"{'전체':14s} {cg:9.3f} {cgt:11.3f} {cg-cgt:7.3f}  (n={len(per)})")
    for name, lo, hi in [("단문(1~3)", 1, 3), ("중문(4~8)", 4, 8), ("장문(9~15)", 9, 99)]:
        sub = [p for p in per if lo <= p[0] <= hi]
        if sub:
            cg, cgt = agg(sub)
            lines.append(f"{name:14s} {cg:9.3f} {cgt:11.3f} {cg-cgt:7.3f}  (n={len(sub)})")
    report = "\n".join(lines)
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
