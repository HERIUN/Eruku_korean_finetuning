"""echo CER(문자오류율) 평가 — OCR 로 생성이미지를 읽어 target 과 비교(정렬무관·언어공정).

MSE/SSIM 은 공간정렬 오차에 민감하고 언어간 비교가 불공정. CER 은 텍스트 정확도를 직접 재고
누락/반복(완성도 실패)을 그대로 벌하므로 "target 을 정확·끝까지 렌더" 능력의 신뢰지표.
OCR 자체 오차 보정: GT 렌더의 CER(OCR 바닥)도 같이 재서 gen 과의 gap 을 봄.

CER = edit_distance(OCR읽음, target) / len(target). 공백 정규화.
easyocr(ko+en). N(기본 100)개 다양길이, 길이버킷 분해.

사용: CUDA_VISIBLE_DEVICES=2 python eval_cer.py --ckpt <ckpt> --n 100 [--english] --out _val/cer.txt
"""
from __future__ import annotations
import argparse, random, sys, re
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from infer_show import load_model, gen_arr, render_in_font
from eval_echo_metrics import build_texts, style_tensor, font_can_render


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def cer(hyp, ref):
    hyp, ref = norm(hyp), norm(ref)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein (char)
    m, n = len(hyp), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (hyp[i - 1] != ref[j - 1]))
            prev = cur
    return dp[n] / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vae-checkpoint", default=None,
                    help="한글 적응 VAE(vae_sXXXX) 로 교체 평가. 주면 ckpt 의 옛 vae.* 를 strip")
    ap.add_argument("--fonts-dir", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--max-words", type=int, default=15, help="문장 최대 어절수(1~max, 올리면 긴 문장 robustness)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--english", action="store_true")
    ap.add_argument("--coherent", action="store_true", help="랜덤단어 대신 실제 문장 세그먼트(OCR 신뢰성↑)")
    ap.add_argument("--no-style-text", action="store_true")
    ap.add_argument("--ocr-gpu", action="store_true", help="OCR 를 GPU 로(기본 CPU, 학습과 GPU 경합 회피)")
    ap.add_argument("--out", default=str(HERE / "finetune_runs" / "cer.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    import easyocr
    reader = easyocr.Reader(['ko', 'en'], gpu=args.ocr_gpu, verbose=False)
    # OCR feature: recognizer 의 SequenceModeling(BiLSTM 문맥특징) 출력을 hook 으로 캡처.
    # gen/GT 이미지의 이 특징을 코사인 비교 → 정렬무관·언어공정 유사도(CER 의 OCR바닥 문제 우회).
    _feat = {}
    def _hook(m, i, o):
        t = o[0] if isinstance(o, (tuple, list)) else o
        _feat['v'] = t.detach().float().reshape(-1, t.shape[-1]).mean(0)   # [C]
    # --ocr-gpu 면 recognizer 가 DataParallel 로 감싸져 .module 아래에 있음
    _rec = reader.recognizer.module if hasattr(reader.recognizer, "module") else reader.recognizer
    _rec.SequenceModeling.register_forward_hook(_hook)

    def ocr(arr):
        # arr: grayscale uint8 [H,W] → 텍스트 + OCR feature([C])
        _feat.clear()
        res = reader.readtext(np.stack([arr] * 3, -1), detail=0, paragraph=True)
        return (" ".join(res) if res else ""), _feat.get('v')

    model, step = load_model(args.ckpt, device, vae_checkpoint=args.vae_checkpoint)
    fonts = sorted(p for p in Path(args.fonts_dir).glob("*.ttf"))
    texts = build_texts(args.n, rng, english=args.english, coherent=args.coherent, max_words=args.max_words)

    per = []   # (nw, cer_gen, cer_gt)
    done = 0
    for text in texts:
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text)), None)
        if fp is None:
            continue
        gt = render_in_font(fp, text)
        st = style_tensor(gt).unsqueeze(0).to(device)
        mi = model.get_model_inputs([st[0]], None, style_len=st.shape[-1], gen_len=None, max_img_len=2048)
        dec = mi["decoder_inputs_embeds"].to(device); spx = dec.shape[1] * 8
        stext = "" if args.no_style_text else text
        try:
            gen = gen_arr(model, dec, stext, text, args.cfg, args.max_new_tokens, spx, device)
        except Exception as e:
            print(f"  [skip] {e}"); continue
        tg, fg = ocr(gen)
        tgt, fgt = ocr(gt)
        cg = cer(tg, text); cgt = cer(tgt, text)
        sim = (float(torch.cosine_similarity(fg, fgt, dim=0))
               if (fg is not None and fgt is not None and fg.shape == fgt.shape) else float('nan'))
        per.append((len(text.split()), cg, cgt, sim)); done += 1
        if done % 20 == 0:
            print(f"  {done}/{args.n} ... (누적 gen CER {np.mean([p[1] for p in per]):.3f})")

    def agg(rows):
        return (np.mean([p[1] for p in rows]), np.mean([p[2] for p in rows]),
                np.nanmean([p[3] for p in rows]))
    lines = [f"=== echo CER+OCRfeat 평가 (step {step}, n={done}, cfg={args.cfg}, "
             f"{'영어' if args.english else '한글'}{', no-stext' if args.no_style_text else ''}) ===",
             f"fonts-dir: {args.fonts_dir}   (CER↓ gen=생성 GT=OCR바닥 / OCRsim↑ gen-GT 특징 코사인)",
             f"{'':14s} {'gen CER↓':>9} {'GT CER':>8} {'gap':>7} {'OCRsim↑':>8}"]
    cg, cgt, sim = agg(per)
    lines.append(f"{'전체':14s} {cg:9.3f} {cgt:8.3f} {cg-cgt:7.3f} {sim:8.3f}  (n={len(per)})")
    for name, lo, hi in [("단문(1~3)", 1, 3), ("중문(4~8)", 4, 8), ("장문(9~15)", 9, 99)]:
        sub = [p for p in per if lo <= p[0] <= hi]
        if sub:
            cg, cgt, sim = agg(sub)
            lines.append(f"{name:14s} {cg:9.3f} {cgt:8.3f} {cg-cgt:7.3f} {sim:8.3f}  (n={len(sub)})")
    report = "\n".join(lines)
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
