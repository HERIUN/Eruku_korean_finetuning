"""§13 2단계 진단: FLUX VAE Eruku 의 배경 회색화·조기 종료·흐림이 어디서 오나.

같은 (폰트 style, 텍스트) 를 세 경로로 디코딩해 비교한다.
  gt  : 정답 이미지(같은 폰트로 렌더)를 VAE 왕복 — 상한
  tf  : teacher forcing — 매 열이 **정답** 이전 열을 보고 낸 1-step 예측(학습과 같은 조건)
  gen : 자유 생성 — 자기 예측을 다시 입력(추론 조건)
tf 가 깨끗하고 gen 만 나쁘면 자기회귀 오차 누적(exposure bias), tf 도 나쁘면 1-step 회귀 자체의 한계.
조기 종료: tf 에서 정답 끝 전 위치들의 EOG 확률 최대값, gen 폭 / 정답 폭.

  CUDA_VISIBLE_DEVICES=2 .venv/bin/python experiments/flux_diag.py --ckpt <ckpt> [--save-dir <dir>]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "experiments"))
from infer.show import load_model, style_tensor, gen_from_style, render_in_font

STYLE_T = "조용히 책을 읽었다"
TEXTS = ["한국어 손글씨 생성", "밝은 햇살 붉은 노을이 강을 물들였다",
         "형형색색 꽃들이 활짝 피어난 봄날의 공원에서 아이들이 뛰어놀았다"]


EN_STYLE_T = "I read a book quietly"
EN_TEXTS = ["Handwriting generation", "The quick brown fox jumps over the lazy dog",
            "Bright sunlight painted the river red at dusk while children played"]


def fonts(lang="ko"):
    if lang == "en":
        return [R / "assets/fonts_english" / f"{f}.ttf" for f in ("Caveat", "IndieFlower", "Kalam", "DancingScript")]
    t = [R / "assets/fonts_korean_v2/test" / f"{f}.ttf" for f in ("NanumPenScript", "Gaegu", "HiMelody", "Dokdo")]
    return t + sorted((R / "assets/custom_hw_fonts").glob("*.otf"))


def u8(x):   # [1,h,w] or [h,w] [-1,1] → uint8
    return ((x.squeeze().clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def bg_bins(im, n=5):
    """열별 배경(90퍼센타일)을 n 구간 평균."""
    bg = np.percentile(im.astype(np.float32), 90, axis=0)
    idx = np.linspace(0, len(bg), n + 1).astype(int)
    return [bg[a:b].mean() for a, b in zip(idx[:-1], idx[1:])]


def gray(im):   # 회색(30~230) 픽셀 비율 — 흐림/안개 지표
    return float(((im >= 30) & (im < 230)).mean())


@torch.no_grad()
def main(a):
    dev = torch.device("cuda")
    global STYLE_T, TEXTS
    if a.lang == "en":
        STYLE_T, TEXTS = EN_STYLE_T, EN_TEXTS
    m, step = load_model(a.ckpt, dev)
    m.dropout_probability = 0.0                     # forward 의 텍스트/이미지 dropout 끄기
    cap = {}
    m.t5_to_special.register_forward_hook(lambda mod, i, o: cap.__setitem__("sp", o))
    rows = {k: [] for k in ("gt", "tf", "gen")}; eog_max, eog_end, width = [], [], []
    for fi, fp in enumerate(fonts(a.lang)):
        sty = style_tensor(render_in_font(fp, STYLE_T)).to(dev)
        for ti, t in enumerate(TEXTS):
            g = style_tensor(render_in_font(fp, t)).to(dev)
            mi = m.get_model_inputs([sty], [g], [sty.shape[-1]], [g.shape[-1]], 4096)
            z, sp = mi["decoder_inputs_embeds"], mi["specials"]
            _, pred = m(z, sp, [STYLE_T], [t])      # pred: [1,c,8,T], 열 i = 입력 열 i 의 예측
            p_eog = cap["sp"].softmax(-1)[0, :, 1]  # 열 i 가 EOG 일 확률
            s = sp[0]; sog = int((s == 0).nonzero()[0]); eog = int((s == 1).nonzero()[0])
            gen_cols = slice(sog + 1, eog)
            gt_img = u8(m._vae_decode(m.z_rearrange(z[:, gen_cols])))
            tf_img = u8(m._vae_decode(pred[..., gen_cols]))
            fr_img = gen_from_style(m, sty, STYLE_T, t, 1.0, 480, dev, seed=0)
            for k, im in (("gt", gt_img), ("tf", tf_img), ("gen", fr_img)):
                rows[k].append((ti, bg_bins(im), gray(im), im))
            eog_max.append(float(p_eog[sog + 1:eog - 2].max()))   # 정답 끝 2열 전까지 중 최대 EOG 확률
            eog_end.append(float(p_eog[eog]))
            width.append(fr_img.shape[1] / gt_img.shape[1])
    print(f"=== {a.ckpt}  s{step}  (폰트 {len(fonts(a.lang))} × 텍스트 {len(TEXTS)})")
    for k in ("gt", "tf", "gen"):
        for ti in range(len(TEXTS)):
            r = [x for x in rows[k] if x[0] == ti]
            print(f"  {k:3s} {len(TEXTS[ti]):2d}자  배경 " + " → ".join(f"{v:.0f}" for v in np.mean([x[1] for x in r], 0))
                  + f"   회색비율 {np.mean([x[2] for x in r]):.3f}")
    w = np.array(width)
    print(f"  EOG 확률: 정답 끝 위치 평균 {np.mean(eog_end):.3f} | 끝 이전 최대값 평균 {np.mean(eog_max):.3f}, >0.5 인 샘플 {np.mean(np.array(eog_max) > 0.5):.0%}")
    print(f"  생성 폭/정답 폭: 중앙값 {np.median(w):.2f}, <0.8(조기 종료) {np.mean(w < 0.8):.0%}, >1.3(폭주) {np.mean(w > 1.3):.0%}")
    if a.save_dir:   # 텍스트 2(가장 긴 것) 의 gt/tf/gen 을 폰트별로 쌓아 저장
        out = Path(a.save_dir); out.mkdir(parents=True, exist_ok=True)
        ims = [x[3] for k in ("gt", "tf", "gen") for x in rows[k] if x[0] == 2]
        n = len(fonts(a.lang)); ims = [ims[i + j * n] for i in range(n) for j in range(3)]
        W = max(i.shape[1] for i in ims)
        Image.fromarray(np.concatenate([np.pad(i, ((0, 4), (0, W - i.shape[1])), constant_values=255) for i in ims])
                        ).save(out / f"diag_s{step}_{a.lang}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--save-dir", default="")
    ap.add_argument("--lang", choices=["ko", "en"], default="ko")
    main(ap.parse_args())
