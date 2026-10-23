"""IAM 공식 벤치마크(SHTG-IAM lines/words, HWD 패키지) — Eruku 논문 Table 3 프로토콜.

우리 폰트셋 평가(echo_metrics/htr_cer)는 style 이 폰트 렌더라 '실제 필기' 일반화를 못 잰다.
여기선 IAM test 필기자(gan.iam.test.gt.filter27)의 **실제 손글씨 라인**을 style 로 주고,
같은 필기자의 다른 라인 텍스트를 생성 → 그 필기자의 진짜 라인과 비교한다.
  FID/BFID/KID : 생성 vs 실제 분포 (↓). **unfold()(라인 전체를 정사각 패치로)** 후 height 32 —
                 기본(첫 정사각만 crop)으로 재면 논문과 안 맞는다. 공식 Emuru 생성물(GeneratedDataset
                 'iam_lines__emuru')로 검증: unfold 13.90/6.18 = 논문 13.89/6.19, crop 30.6/18.2.
  HWD          : 필기자별 VGG16 특징거리 (↓, 스타일 충실도, 논문과 일치 1.87)
  ΔCER         : TrOCR CER(생성) − CER(실제) (↓, 논문 지표). 공식 Emuru 0.226−0.082=0.14 = 논문 0.14.
논문 Eruku 는 cfg γ=1.25 (lines: HWD 1.70 ΔCER 0.06 FID 16.40 BFID 4.88).
영어 전용 벤치마크다 — 한글 성능은 여기서 안 나온다.
데이터는 hwd 가 ~/.cache/iam 에 자동으로 받는다.

--protocol upstream : 원본 repo(Blowing-Up-Groundhogs/Eruku) generate_iam_eruku.py 를 그대로 따른다.
  ① 생성 길이 상한 = 정답 이미지 폭(높이 64 환산)/8 × 1.5 토큰 (정답 길이를 미리 앎)
  ② style 폭 뒤를 자르고 trim_white(앞쪽 style 잔여획·앞뒤 흰 여백 제거, 8px 패딩)
  ③ 저장은 [-1,1] 출력을 그대로 save_image → [0,1] 로 clamp (중간 회색이 검게 눌림)
  ④ style LANCZOS 리사이즈, words 는 IAMWordsFromLines(style = 같은 필기자의 '라인')
  원본은 cfg 를 안 넘기지만(=1.0) 논문 기본 γ=1.25 라 --cfg 는 그대로 받는다.

사용:
  CUDA_VISIBLE_DEVICES=2 python eval/iam_shtg.py --ckpt <ckpt> --split lines --cfg 1.25 --out finetune_runs/_iam/<label>
  (--n 500 으로 부분집합 — 필기자 고르게 등간격 추출. 기본 전체)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
sys.path.insert(0, str(HERE))

from infer.show import load_model, style_tensor, gen_from_style_batch, DEFAULT_MAX_IMG_LEN


def trim_white(img, threshold=0.9, padding=8):
    """원본 generate_iam_eruku.trim_white 그대로 ([c,h,w], [-1,1] 공간에서 threshold)."""
    start, end = 0, img.size(-1)
    vmin = img[0].min(-2).values.tolist()
    for v in vmin:            # 앞쪽 비흰 열(style 잔여획) 건너뜀
        if v >= threshold: break
        start += 1
    for v in vmin:            # 이어지는 흰 열 건너뜀 (원본도 처음부터 다시 센다)
        if v < threshold: break
        start += 1
    for v in vmin[::-1]:      # 뒤쪽 흰 열
        if v < threshold: break
        end -= 1
    start, end = max(start - padding, 0), min(end + padding, img.size(-1))
    return img if start >= end else img[..., start:end]


def gen_upstream(model, items, args, dev, seed):
    """원본 프로토콜 배치판: 샘플별 상한 = 정답폭×1.5 토큰 → 넘친 부분은 잘라서 원본(샘플별 max_new)과 같게."""
    styles = [style_tensor(np.array(it["style_imgs"][0].convert("L")), resample=Image.LANCZOS).to(dev) for it in items]
    tgt = [int(it["gen_img"].size[0] * 64 / it["gen_img"].size[1] / 8 * 1.5) for it in items]
    mi = model.get_model_inputs(styles, None, style_len=[el.shape[-1] for el in styles],
                                gen_len=None, max_img_len=args.max_img_len)
    torch.manual_seed(seed)
    imgs, _ = model.generate_batch(mi["decoder_inputs_embeds"].to(dev),
                                   ["" if args.no_style_text else it["style_imgs_text"][0] for it in items],
                                   [it["gen_text"] for it in items], cfg_scale=args.cfg,
                                   max_new_tokens=max(tgt), prefix_lens=mi["lengths"])
    out = []
    for im, n, st in zip(imgs, tgt, styles):
        w = st.shape[-1]                                   # 원본: result[..., img.size(-1):]
        g = trim_white(im[:, :, w:w + n * 8])
        out.append((g.clamp(0, 1) * 255).round().byte()[0].cpu().numpy())   # save_image 와 같은 [0,1] clamp
    return out


def generate(args, ds, fake):
    idx = list(range(len(ds)))
    if args.n and args.n < len(idx):
        idx = idx[::len(idx) // args.n][:args.n]
    todo = [i for i in idx if not (fake / ds.data[i]["dst"]).exists()]   # 이어 돌리기
    todo.sort(key=lambda i: len(ds.data[i]["text"]))                     # 길이 비슷한 것끼리 배치
    print(f"{len(idx)}샘플 중 {len(todo)}개 생성", flush=True)
    if not todo:   # 이미 다 있으면 채점만 (다른 생성기로 만든 fake/ 채점에도 씀)
        return idx
    dev = torch.device("cuda")
    model, step = load_model(args.ckpt, dev)
    for b in range(0, len(todo), args.batch):
        items = [ds[i] for i in todo[b:b + args.batch]]
        if args.protocol == "upstream":
            gens = gen_upstream(model, items, args, dev, args.seed * 1_000_003 + b)
            for it, g in zip(items, gens):
                (fake / it["dst_path"]).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(g).save(fake / it["dst_path"])
            print(f"  {b + len(items)}/{len(todo)}", flush=True)
            continue
        styles = [style_tensor(np.array(it["style_imgs"][0].convert("L"))) for it in items]
        stexts = ["" if args.no_style_text else it["style_imgs_text"][0] for it in items]   # w/o T_s = 빈 문자열
        gens = gen_from_style_batch(model, styles, stexts,
                                    [it["gen_text"] for it in items], args.cfg, args.max_new_tokens,
                                    dev, max_img_len=args.max_img_len, seed=args.seed * 1_000_003 + b)
        for it, g in zip(items, gens):
            (fake / it["dst_path"]).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(g).save(fake / it["dst_path"])
        print(f"  {b + len(items)}/{len(todo)}", flush=True)
    del model; torch.cuda.empty_cache()
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="로컬 .pth/.bin 또는 HF repo id")
    ap.add_argument("--out", required=True, help="결과 폴더 (fake/ real/ report.txt)")
    ap.add_argument("--split", choices=["lines", "words"], default="lines")
    ap.add_argument("--n", type=int, default=0, help="0 = 전체")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=1.25, help="논문 Eruku 기본 γ=1.25")
    ap.add_argument("--max-new-tokens", type=int, default=480)
    ap.add_argument("--max-img-len", type=int, default=DEFAULT_MAX_IMG_LEN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", choices=["ours", "upstream"], default="ours",
                    help="upstream = 원본 generate_iam_eruku.py (정답폭×1.5 상한·trim_white·IAMWordsFromLines)")
    ap.add_argument("--no-style-text", action="store_true", help="w/o T_s — style 전사 안 줌 (논문 Table 6)")
    args = ap.parse_args()

    from hwd.datasets.shtg import IAMLines, IAMWords, IAMWordsFromLines
    from hwd.datasets import FolderDataset
    from hwd.scores import FIDScore, BFIDScore, KIDScore, HWDScore, CERScore

    up = args.protocol == "upstream"
    if args.split == "lines":
        ds = IAMLines(num_style_samples=1, load_gen_sample=up)
    else:
        ds = (IAMWordsFromLines if up else IAMWords)(num_style_samples=1, load_gen_sample=up)
    out = Path(args.out); fake, real = out / "fake", out / "real"
    idx = generate(args, ds, fake)

    # 정답 = 생성한 것과 같은 샘플만 (부분집합일 때 분포 맞춤)
    full = ds.data
    ds.data = [full[i] for i in idx]
    if not ds.check_compliance(real):
        ds.save_reference(real)
    ds.save_transcriptions(fake); ds.save_transcriptions(real)
    ds.data = full

    F, R = FolderDataset(str(fake)), FolderDataset(str(real))
    res = {"n": len(F), "writers": len(F.author_ids)}
    res["HWD"] = float(HWDScore(height=32)(F, R))
    cer = CERScore(height=64)
    res["CER"] = float(cer(F)); res["CER_real"] = float(cer(R)); res["dCER"] = res["CER"] - res["CER_real"]
    Fu, Ru = FolderDataset(str(fake)).unfold(), FolderDataset(str(real)).unfold()
    for name, sc in (("FID", FIDScore(height=32)), ("BFID", BFIDScore(height=32)), ("KID", KIDScore(height=32))):
        res[name] = float(sc(Fu, Ru)); print(name, res[name], flush=True)

    report = (f"=== SHTG-IAM {args.split}  ckpt={args.ckpt}  n={res['n']} writers={res['writers']} cfg={args.cfg}{' w/o T_s' if args.no_style_text else ''} protocol={args.protocol}\n"
              f"HWD {res['HWD']:.2f}  ΔCER {res['dCER']:.2f}  FID {res['FID']:.2f}  BFID {res['BFID']:.2f}  "
              f"KID {res['KID']:.4f}   (CER 생성 {res['CER']:.4f} / 실제 {res['CER_real']:.4f})")
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    (out / "report.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
