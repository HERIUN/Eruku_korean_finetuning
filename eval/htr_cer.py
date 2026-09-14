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

from configs import loader as _cfgloader
from infer.show import load_model, render_in_font, style_tensor, gen_from_style
from eval.echo_metrics import build_texts, font_can_render
from train.aux_htr import load_pretrained_htr, cer as char_cer
from custom_datasets.korean.alphabet import get_korean_alphabet
from custom_datasets.upstream.subsequent_mask import subsequent_mask
from custom_datasets.korean.aux import _fit_width
from experiments.common import load_line_x11, roundtrip, to_u8

ALPHA = get_korean_alphabet()


def to_htr_input(arr, dev, binarize=0):
    """grayscale uint8 [H,W](ink~0 bg~255) → [1,1,64,768] [-1,1] (HTR 입력).

    binarize>0 이면 그 임계값으로 이진화한다. VAE 복원은 글자 뒤 꼬리에 연회색 띠(≈237)를
    남길 수 있는데, MSE 로는 안 잡히지만 HTR 리더가 그걸 `[ ] " ' ( )` 로 읽어 짧은 목표에서
    CER 을 1~3 까지 튀긴다(2026-09-09 실측). 이진화하면 그 오탐이 사라진다 —
    GT 바닥은 거의 안 변하고(0.009→0.007) 폭주율은 0% 가 된다."""
    if binarize:
        arr = np.where(arr < binarize, 0, 255).astype(np.uint8)
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
    ap.add_argument("--ckpt", required=True,
                    help="로컬 .pth 경로 또는 HF repo id (예: HERIUN/eruku_korean)")
    ap.add_argument("--vae-checkpoint", default=None)
    ap.add_argument("--htr-checkpoint", default="finetune_runs/aux_htr_ko/htr_s20000")
    ap.add_argument("--fonts-dir", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--max-words", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--english", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--coherent", action=argparse.BooleanOptionalAction, default=False, help="랜덤단어 대신 실제 문장(권장)")
    ap.add_argument("--binarize", type=int, default=0, metavar="TH",
                    help="리더 입력을 이 임계값으로 이진화(0=끔, 권장 200). VAE 복원의 연회색 "
                         "꼬리를 리더가 구두점으로 오독하는 것을 막는다. 과거 수치와 비교하려면 끌 것")
    ap.add_argument("--vae-only", action=argparse.BooleanOptionalAction, default=False,
                    help="T5 생성 대신 GT 를 VAE 로만 왕복(encode→decode)시켜 읽는다 = 이 VAE 로 "
                         "도달 가능한 CER 하한. 어떤 T5 도 이보다 좋아질 수 없다(docs/EXPERIMENTS.md §3)")
    ap.add_argument("--no-style-text", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--style-ref-text", default=None,
                    help="style ref 를 이 텍스트의 렌더로(one-shot 프로토콜, 예: gen_compare.STYLE_TEXT_KO). "
                         "기본(None)=echo(style = 목표 텍스트 렌더) — InkSpire 엔 복사 과제가 되므로 둘 다 측정할 것")
    ap.add_argument("--out", default=str(HERE / "finetune_runs" / "htr_cer.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-img-len", type=int, default=8192,
                    help="style prefix 예산(px). style 은 이 값의 절반까지만 쓰인다. "
                         "학습 run 의 --max-img-len 과 맞춰야 조건 분포가 어긋나지 않음")
    ap.add_argument("--config", default=str(HERE / "configs/eval.yaml"),
                    help="설정 yaml. 우선순위 CLI > yaml > 코드 기본값 (configs/README.md)")
    args = _cfgloader.parse_args(ap, default_config=str(HERE / "configs/eval.yaml"), scopes=('common', 'htr_cer'))
    dev = torch.device(args.device)
    rng = random.Random(args.seed)

    model, step = load_model(args.ckpt, dev, vae_checkpoint=args.vae_checkpoint)
    htr = load_pretrained_htr(Path(args.htr_checkpoint)).to(dev).eval()
    assert htr.fc.out_features == len(ALPHA), "HTR alphabet 불일치"
    fonts = sorted(p for p in Path(args.fonts_dir).glob("*.[ot]tf"))   # custom_hw_fonts 는 otf
    sref = args.style_ref_text
    texts = build_texts(args.n, rng, english=args.english, coherent=args.coherent, max_words=args.max_words)

    per = []  # (nw, cer_gen, cer_gt, ink, font_stem)
    done = 0
    for i_t, text in enumerate(texts):
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text) and (not sref or font_can_render(f, sref))), None)
        if fp is None:
            continue
        gt = render_in_font(fp, text)
        style = render_in_font(fp, sref) if sref else gt
        stext = "" if args.no_style_text else (sref or text)
        try:
            if args.vae_only:      # 상한 측정 — T5 를 빼고 GT 를 VAE 로만 왕복시킨다
                # ★ 학습과 같은 기하(_fit_width → 폭 768)로 넣어야 한다. 8배수 패딩만 하면
                # 단문이 80~144px 로 들어가는데 VAE 는 768 폭으로만 학습돼(aux.py:64) 좁은 폭에서
                # 글자 뒤에 연회색 띠를 남긴다(244 vs 254). 그 띠를 HTR 이 구두점으로 오독해
                # 단문 CER 이 1~3 까지 튀었다 — VAE 품질이 아니라 평가 기하 불일치였다(2026-09-09).
                w64 = max(1, int(round(64 * gt.shape[1] / gt.shape[0])))
                x = _fit_width(load_line_x11(gt)[0]).unsqueeze(0)
                gen = to_u8(roundtrip(model.vae, x, dev)[..., :min(w64, 768)])
            else:
                # ★ eruku.py:168 이 추론에서도 latent_dist.sample() 을 쓴다 → 시드를 안 박으면
                # 같은 모델·같은 텍스트를 다시 재도 CER 이 달라진다(과거 단문 0.280 vs 1.076).
                # 샘플 인덱스로 고정해 run 간 비교가 성립하게 한다.
                gen = gen_from_style(model, style_tensor(style), stext, text,
                                     args.cfg, args.max_new_tokens, dev,
                                     max_img_len=args.max_img_len,
                                     seed=args.seed * 1_000_003 + i_t)
        except Exception as e:
            print(f"  [skip] {e}"); continue
        hyp_gen = htr_read(htr, to_htr_input(gen, dev, args.binarize), dev)[0]
        hyp_gt = htr_read(htr, to_htr_input(gt, dev, args.binarize), dev)[0]
        cg = char_cer([norm(hyp_gen)], [norm(text)])
        cgt = char_cer([norm(hyp_gt)], [norm(text)])
        ink = float((gen < 200).mean())
        per.append((len(text.split()), cg, cgt, ink, fp.stem)); done += 1
        if done % 20 == 0:
            print(f"  {done}/{args.n} ... (누적 gen CER {np.mean([p[1] for p in per]):.3f})")

    # CER 은 위로 열려 있다(runaway 는 3~5 까지 나온다). 표본 몇 건이 평균을 지배하므로
    # 평균만 보면 판정이 뒤집힌다 — 중앙값과 폭주율(CER>1 = 목표보다 긴 출력)을 같이 낸다.
    RUNAWAY = 1.0

    BLANK_INK = 0.02          # 이보다 잉크가 적으면 '생성 실패(백지)' — 폭주와 원인이 다르다

    def agg(rows):
        g = np.array([p[1] for p in rows])
        ink = np.array([p[3] for p in rows])
        return (g.mean(), np.median(g), float((g > RUNAWAY).mean()),
                float((ink < BLANK_INK).mean()), np.mean([p[2] for p in rows]))
    lines = [f"=== HTR-reader CER 평가 (step {step}, n={done}, cfg={args.cfg}, "
             f"{'영어' if args.english else '한글'}{', no-stext' if args.no_style_text else ''}"
             f"{f', style-ref={sref!r}' if sref else ', echo'}"
             f"{f', bin{args.binarize}' if args.binarize else ''}) ===",
             f"reader: 한글 HTR {args.htr_checkpoint} (렌더바닥 ~0.00) | ckpt vae: {args.vae_checkpoint or 'ckpt내장'}",
             f"{'':14s} {'평균CER↓':>9} {'중앙값↓':>8} {'폭주율↓':>8} {'백지율↓':>8} {'GT바닥':>8}"]

    def row(name, rows):
        m, md, ra, bl, cgt = agg(rows)
        lines.append(f"{name:14s} {m:9.3f} {md:8.3f} {ra:7.1%} {bl:7.1%} {cgt:8.3f}  (n={len(rows)})")
    row("전체", per)
    for name, lo, hi in [("단문(1~3)", 1, 3), ("중문(4~8)", 4, 8), ("장문(9~15)", 9, 99)]:
        sub = [p for p in per if lo <= p[0] <= hi]
        if sub:
            row(name, sub)
    # 폰트가 16종·4종뿐인 held-out probe 에서는 폰트 하나가 전체 수치를 흔든다 → 분해해 둔다.
    byf = {}
    for p in per:
        byf.setdefault(p[4], []).append(p)
    if 1 < len(byf) <= 40:
        lines.append("")
        lines.append(f"{'폰트별':14s} {'중앙값↓':>8} {'폭주율↓':>8} {'백지율↓':>8}")
        for f_, rows in sorted(byf.items(), key=lambda kv: -agg(kv[1])[1]):
            _, md, ra, bl, _ = agg(rows)
            lines.append(f"{f_[:14]:14s} {md:8.3f} {ra:7.1%} {bl:7.1%}  (n={len(rows)})")
    report = "\n".join(lines)
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
