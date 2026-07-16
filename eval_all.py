"""통합 평가 배터리 — 한 체크포인트를 全metric(echo MSE/SSIM/LPIPS/FID + CER)·한/영으로 측정.

목표 판정용: pretrained 대비 全metric 우월 + 한글 CER = pretrained-영어 수준 을 한 표로 본다.
학습과 GPU 경합 금지 — 반드시 학습 중단(GPU free) 후 실행.

사용:
  CUDA_VISIBLE_DEVICES=2 python eval_all.py --n 100 \
    --models pretrained=<bin> v2_20k=<pth> v3b=<pth>
각 모델을 한글·영어 echo+CER 로 재고, 표 저장.
"""
from __future__ import annotations
import argparse, subprocess, sys, re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_one(label, ckpt, n, english, kind, coherent=False, fonts_dir=None, max_words=15):
    """kind: 'echo' or 'cer'. 반환: 지표 dict."""
    script = "eval_echo_metrics.py" if kind == "echo" else "eval_cer.py"
    out = HERE / f"finetune_runs/_evalall_{label}_{'en' if english else 'ko'}_{kind}.txt"
    cmd = [".venv/bin/python", script, "--ckpt", ckpt, "--n", str(n),
           "--fonts-dir", str(fonts_dir), "--max-words", str(max_words), "--out", str(out)]
    if english:
        cmd.append("--english")
    if coherent:
        cmd.append("--coherent")
    print(f"  running {kind} {label} {'en' if english else 'ko'} ...", flush=True)
    subprocess.run(cmd, cwd=str(HERE))
    txt = out.read_text(encoding="utf-8") if out.exists() else ""
    m = re.search(r"전체\s+([\d.]+)\s+([\d.]+)\s+([\d.-]+)", txt)
    if kind == "echo":
        # echo 전체행: MSE SSIM LPIPS  (FID/BFID 는 별도 줄)
        vals = re.search(r"전체\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", txt)
        fid = re.search(r"\nFID\(.*?:\s*([\d.-]+)", txt)
        bfid = re.search(r"BFID\(.*?:\s*([\d.-]+)", txt)
        return {"MSE": vals.group(1), "SSIM": vals.group(2), "LPIPS": vals.group(3),
                "FID": fid.group(1) if fid else "-",
                "BFID": bfid.group(1) if bfid else "-"} if vals else {}
    else:
        # CER 전체행: gen GT gap OCRsim
        sim = re.search(r"전체\s+[\d.]+\s+[\d.]+\s+[\d.-]+\s+([\d.-]+)", txt)
        return {"CER": m.group(1), "OCRsim": sim.group(1) if sim else "-"} if m else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="label=ckpt 형식들")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--langs", nargs="+", default=["en", "ko"])
    ap.add_argument("--coherent", action="store_true", help="실제 문장 세그먼트로 평가(신뢰성↑)")
    ap.add_argument("--max-words", type=int, default=15, help="문장 최대 어절수(올리면 긴 문장 테스트)")
    ap.add_argument("--fonts-dir-ko", default=str(HERE / "assets/fonts_korean_v2/train"),
                    help="한글 style 폰트 디렉토리(일반화 보려면 fonts_korean_unseen)")
    ap.add_argument("--fonts-dir-en", default=str(HERE / "assets/fonts_english"),
                    help="영어 style 폰트 디렉토리")
    args = ap.parse_args()
    models = [(m.split("=", 1)[0], m.split("=", 1)[1]) for m in args.models]

    rows = {}
    for label, ckpt in models:
        rows[label] = {}
        for lang in args.langs:
            en = (lang == "en")
            fdir = args.fonts_dir_en if en else args.fonts_dir_ko
            echo = run_one(label, ckpt, args.n, en, "echo", coherent=args.coherent,
                           fonts_dir=fdir, max_words=args.max_words)
            cer = run_one(label, ckpt, args.n, en, "cer", coherent=args.coherent,
                          fonts_dir=fdir, max_words=args.max_words)
            rows[label][lang] = {**echo, **cer}

    # 표 출력
    lines = [f"=== 통합 평가 (n={args.n}, echo+CER, seen 폰트) ===",
             "(MSE·LPIPS·FID·CER ↓ 낮을수록 / SSIM ↑)"]
    for lang in args.langs:
        lines.append(f"\n[{lang.upper()}]  (MSE·LPIPS·FID·BFID·CER↓ / SSIM·OCRsim↑)")
        lines.append(f"{'model':14s} {'MSE':>7} {'SSIM':>7} {'LPIPS':>7} {'FID':>7} "
                     f"{'BFID':>7} {'CER':>7} {'OCRsim':>7}")
        for label, _ in models:
            d = rows[label].get(lang, {})
            lines.append(f"{label:14s} {d.get('MSE','-'):>7} {d.get('SSIM','-'):>7} "
                         f"{d.get('LPIPS','-'):>7} {d.get('FID','-'):>7} {d.get('BFID','-'):>7} "
                         f"{d.get('CER','-'):>7} {d.get('OCRsim','-'):>7}")
    report = "\n".join(lines)
    print("\n" + report)
    op = HERE / "finetune_runs/eval_all_report.txt"
    op.write_text(report, encoding="utf-8")
    print(f"\nsaved {op}")


if __name__ == "__main__":
    main()
