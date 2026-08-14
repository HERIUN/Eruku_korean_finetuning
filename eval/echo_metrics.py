"""echo(style_text == gen_text) 정량 평가 — 생성이미지 vs 정답(같은 폰트·텍스트 렌더) 비교.

style==gen 일 때만 픽셀/구조 지표가 유효하다(모델이 그 글자를 그대로 재현해야 하므로).
정답 = 텍스트 T 를 폰트 F 로 깨끗이 렌더한 이미지. 이를 style 조건으로도 주고, echo 로 생성한 뒤
정답과 비교한다. N(기본 100)개를 **다양한 길이(어절 1~15)**로 샘플.

지표:
  MSE   : 픽셀 평균제곱오차(정규화 0~1) ↓
  SSIM  : 구조적 유사도 ↑ (skimage)
  LPIPS : 지각적 거리 ↓ (lpips, AlexNet)
  FID   : 분포 거리 ↓ (pytorch-fid InceptionV3, 전체 셋 vs 정답 셋)
길이 버킷(단문 1~3 / 중문 4~8 / 장문 9~15)별로도 MSE/SSIM/LPIPS 분해 보고.

사용:
  CUDA_VISIBLE_DEVICES=2 python eval/echo_metrics.py --ckpt <ckpt> --n 100 \
      --fonts-dir assets/fonts_korean_v2/train --out _val/echo_metrics.txt
"""
from __future__ import annotations
import argparse, random, sys, re
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image, ImageFont
from torchvision import transforms as T
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))
from infer.show import load_model, render_in_font, style_tensor, gen_from_style


def h64(a, h=64):
    H, W = a.shape
    return cv2.resize(a, (max(1, int(W * h / H)), h), interpolation=cv2.INTER_AREA)


def pad_to(a, W, val=255):
    h, w = a.shape
    return a[:, :W] if w >= W else np.hstack([a, np.full((h, W - w), val, np.uint8)])


# 코헤런트 영어 코퍼스(실제 문장) — 세그먼트 추출용
EN_COHERENT = """the quick brown fox jumps over the lazy dog near the river bank at sunrise .
machine learning models can generate handwritten text in many different styles and fonts today .
she sold seashells by the seashore while the waves crashed gently against the rocks at dawn .
in the beginning there was nothing but darkness and then a brilliant light appeared over the hills .
the five boxing wizards jump quickly over the wooden fence as the summer storm approaches from the west .
scientists around the world are working hard to understand the deep mysteries of the human brain .
a journey of a thousand miles begins with a single step taken in the right direction today .
the old library on the corner holds thousands of books written by famous authors from every century .
technology continues to change the way people communicate work and live their everyday lives now .
children laughed and played in the park while their parents watched them from a nearby wooden bench .""".split("\n")


def build_texts(n, rng, english=False, coherent=False, max_words=15):
    """다양한 길이(어절 1~max_words)의 라인 n개. english=True 면 영어(pretrained 비교용).
    coherent=True 면 실제 문장에서 세그먼트 추출(OCR 신뢰성↑, CER 덜 비관적).
    max_words 올리면 더 긴 문장으로 길이 robustness 측정."""
    import re as _re
    if coherent:
        base = ([" ".join(_re.findall(r"[A-Za-z0-9']+", s)) for s in EN_COHERENT] if english
                else [ln.strip() for ln in (HERE / "assets/corpus/korean_lines.txt")
                      .read_text(encoding="utf-8").splitlines() if ln.strip()])
        out = []
        for i in range(n):
            nw = 1 + (i % max_words)
            words = rng.choice(base).split()
            if len(words) <= nw:
                seg = words
            else:
                st = rng.randint(0, len(words) - nw)
                seg = words[st:st + nw]
            out.append(" ".join(seg))
        return out
    if english:
        cand = (HERE / "assets/corpus/english_words.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
        words = sorted({w.strip() for w in cand if _re.match(r"^[A-Za-z][A-Za-z'-]*$", w.strip() or "x")})
        nums = lambda: rng.choice(["2024", "98.7%", "4,500", "123", "2024-05-11", "15:30"])
    else:
        lines = (HERE / "assets/corpus/korean_lines.txt").read_text(encoding="utf-8").splitlines()
        words = sorted({w for ln in lines for w in ln.split() if w.strip()})
        nums = lambda: rng.choice(["2024년", "98.7%", "4,500원", "123", "010-1234-5678", "15:30"])
    out = []
    for i in range(n):
        nw = 1 + (i % max_words)                # 어절수 1~max_words 균등 분포
        toks = [rng.choice(words) for _ in range(nw)]
        if nw >= 3 and rng.random() < 0.3:      # 가끔 숫자 섞기
            toks[rng.randrange(nw)] = nums()
        out.append(" ".join(toks))
    return out


def font_can_render(fp, text):
    try:
        cm = TTFont(str(fp)).getBestCmap()
    except Exception:
        return False
    return all(ord(c) in cm for c in text if not c.isspace())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fonts-dir", default=str(HERE / "assets/fonts_korean_v2/train"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--max-words", type=int, default=15, help="문장 최대 어절수(1~max, 올리면 긴 문장 robustness)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-style-text", action="store_true")
    ap.add_argument("--english", action="store_true", help="영어 전용 텍스트로 평가(pretrained 공정비교용)")
    ap.add_argument("--coherent", action="store_true", help="랜덤단어 대신 실제 문장 세그먼트(OCR 신뢰성↑)")
    ap.add_argument("--out", default=str(HERE / "finetune_runs" / "echo_metrics.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    model, step = load_model(args.ckpt, device)
    import lpips as lpips_lib
    from skimage.metrics import structural_similarity as ssim_fn
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
    lpips_net = lpips_lib.LPIPS(net='alex', verbose=False).to(device).eval()
    incep = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    fonts = sorted(p for p in Path(args.fonts_dir).glob("*.ttf"))
    texts = build_texts(args.n, rng, english=args.english, coherent=args.coherent, max_words=args.max_words)

    feats_gt, feats_gen = [], []
    # BFID(HWD 공식): Otsu 이진화 후 FID — 배경 무시, 폰트충실도. gen/gt 를 임시폴더에 저장 후 일괄 계산.
    import shutil, tempfile
    bdir = Path(tempfile.mkdtemp(prefix="bfid_"))
    (bdir / "fake" / "w").mkdir(parents=True); (bdir / "real" / "w").mkdir(parents=True)
    per = []   # (nw, mse, ssim, lpips)
    done = 0
    for text in texts:
        # 이 텍스트를 렌더 가능한 폰트 하나 고르기(토푸 방지)
        rng.shuffle(fonts)
        fp = next((f for f in fonts if font_can_render(f, text)), None)
        if fp is None:
            continue
        gt = render_in_font(fp, text)                          # 정답(깨끗 렌더)
        stext = "" if args.no_style_text else text
        try:
            gen = gen_from_style(model, style_tensor(gt), stext, text,
                                 args.cfg, args.max_new_tokens, device)
        except Exception as e:
            print(f"  [skip] {text[:20]!r}: {e}"); continue
        # 정렬: height 64, 공통 폭 패딩
        g_gt, g_gen = h64(gt), h64(gen)
        W = max(g_gt.shape[1], g_gen.shape[1])
        g_gt, g_gen = pad_to(g_gt, W), pad_to(g_gen, W)
        mse = float(((g_gt / 255. - g_gen / 255.) ** 2).mean())
        ss = float(ssim_fn(g_gt, g_gen, data_range=255))
        with torch.no_grad():
            t_gt = style_tensor(g_gt).unsqueeze(0).to(device)   # [-1,1] 3ch
            t_gen = style_tensor(g_gen).unsqueeze(0).to(device)
            lp = float(lpips_net(t_gt, t_gen).item())
            # FID 특징 (Inception, [0,1] 3ch)
            def feat(a):
                x = torch.from_numpy(a / 255.).float()[None, None].repeat(1, 3, 1, 1).to(device)
                return incep(x)[0].reshape(-1).cpu().numpy()   # [2048] 1D
            feats_gt.append(feat(g_gt)); feats_gen.append(feat(g_gen))
        nw = len(text.split())
        per.append((nw, mse, ss, lp)); done += 1
        Image.fromarray(g_gen).convert("RGB").save(bdir / "fake" / "w" / f"{done}.png")
        Image.fromarray(g_gt).convert("RGB").save(bdir / "real" / "w" / f"{done}.png")
        if done % 20 == 0:
            print(f"  {done}/{args.n} ...")

    # per: list of (nw, mse, ssim, lpips). FID = Inception 특징 분포거리.
    def stats(f):
        f = np.array(f).reshape(len(f), -1); return f.mean(0), np.cov(f, rowvar=False)
    mu_g, s_g = stats(feats_gt); mu_p, s_p = stats(feats_gen)
    fid = calculate_frechet_distance(mu_p, s_p, mu_g, s_g)
    # BFID(HWD 공식): Otsu 이진화 후 FID — 폰트충실도(배경/텍스처 무시)
    try:
        from hwd.datasets import FolderDataset
        from hwd.scores import BFIDScore
        bfid = float(BFIDScore(height=32)(FolderDataset(str(bdir / "fake")),
                                          FolderDataset(str(bdir / "real"))))
    except Exception as e:
        bfid = float("nan"); print(f"  BFID 실패: {e}")
    shutil.rmtree(bdir, ignore_errors=True)

    def agg(rows):
        return (np.mean([p[1] for p in rows]), np.mean([p[2] for p in rows]), np.mean([p[3] for p in rows]))
    lines = [f"=== echo 정량평가 (step {step}, n={done}, cfg={args.cfg}, "
             f"{'영어' if args.english else '한글'}"
             f"{', no-stext' if args.no_style_text else ''}) ===",
             f"fonts-dir: {args.fonts_dir}",
             f"{'':14s} {'MSE↓':>8} {'SSIM↑':>8} {'LPIPS↓':>8}"]
    m, s, l = agg(per)
    lines.append(f"{'전체':14s} {m:8.4f} {s:8.4f} {l:8.4f}  (n={len(per)})")
    for name, lo, hi in [("단문(1~3)", 1, 3), ("중문(4~8)", 4, 8), ("장문(9~15)", 9, 99)]:
        sub = [p for p in per if lo <= p[0] <= hi]
        if sub:
            m, s, l = agg(sub)
            lines.append(f"{name:14s} {m:8.4f} {s:8.4f} {l:8.4f}  (n={len(sub)})")
    lines.append(f"\nFID(전체 {done}샘플, 생성 vs 정답): {fid:.2f}")
    lines.append(f"BFID(Otsu이진화 후 FID, 폰트충실도·배경무시): {bfid:.2f}")
    report = "\n".join(lines)
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
