"""PP-OCRv6 한국어 인쇄체+손글씨 인식기(HERIUN/PP-OCRv6-medium-rec-ko-printed-handwriting)로 CER.

★ 프로젝트 .venv 가 아니라 **paddlepaddle-gpu==3.3.1 정확히** 깔린 env 로 돌린다.
  이 export 그래프는 3.0.0('strides' Int32Attribute) / 3.3.0(kernel output defs) /
  3.4.0('epsilon' DoubleAttribute) 전부 로드 실패, 3.3.1 만 된다(2026-10-20 실측). 예:
    uv venv -p 3.12 .venv-paddle && uv pip install -p .venv-paddle/bin/python paddlepaddle-gpu==3.3.1 \
      opencv-python-headless pyyaml --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
  저장소에는 이걸 .venv-paddle 로 만들어 둔다(gitignore). 실행: .venv-paddle/bin/python eval/ppocr_rec.py <폴더>... [--save-preds]
폴더마다 transcriptions.json(상대경로→정답, hwd save_transcriptions 형식)이 있어야 한다.

paddlex 3.4.1 에 PP-OCRv6 가 등록돼 있지 않고 TextRecPredictor 의존성(bidi, pdf…)도 줄줄이
빠져 있어서, paddle.inference 로 직접 돌린다. 전처리·디코딩은 paddlex text_recognition
processors 를 그대로 옮긴 것: BGR, 높이 48 비율유지 리사이즈, (x/255-0.5)/0.5,
CTC greedy(중복제거, blank=0), 문자표 = ['blank'] + inference.yml character_dict + [' '].
CER(corpus) = 전체 편집거리 합 / 전체 정답 글자수 (torchmetrics CharErrorRate 와 같음, IAM 용).
한글 평가는 eval/htr_cer.py 와 같은 판정법(샘플별 CER 의 평균·중앙값·폭주율 CER>1)도 같이 낸다.
공백은 양쪽 다 정규화(htr_cer.norm 과 동일).
  한글 생성 평가: python eval/htr_cer.py --ckpt <ckpt> --coherent --dump-dir <dir>
               .venv-paddle/bin/python eval/ppocr_rec.py <dir>/gt <dir>/gen   (gt = 렌더 바닥)
"""
import argparse, json, math, re, sys
from pathlib import Path
import cv2, numpy as np, yaml
import paddle.inference as pi

REPO = "HERIUN/PP-OCRv6-medium-rec-ko-printed-handwriting"


def model_dir():
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(REPO))
    except ImportError:   # paddle env 에 hf_hub 가 없으면 HF 캐시에서 (먼저 .venv 로 한 번 받아둘 것)
        return sorted(Path("~/.cache/huggingface/hub").expanduser()
                      .glob(f"models--{REPO.replace('/', '--')}/snapshots/*"))[-1]


def load(device_id=0):
    d = model_dir()
    chars = ["blank"] + yaml.safe_load((d / "inference.yml").read_text())["PostProcess"]["character_dict"] + [" "]
    cfg = pi.Config(str(d / "inference.json"), str(d / "inference.pdiparams"))
    cfg.enable_use_gpu(1024, device_id)
    cfg.disable_glog_info()
    pred = pi.create_predictor(cfg)
    return pred, chars


def read(pred, chars, path):
    img = cv2.imread(str(path))                                   # BGR
    h, w = img.shape[:2]
    W = min(3200, max(1, math.ceil(48 * w / h)))
    x = cv2.resize(img, (W, 48)).astype("float32").transpose(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5)[None]
    inp = pred.get_input_handle(pred.get_input_names()[0])
    inp.reshape(x.shape); inp.copy_from_cpu(x)
    pred.run()
    idx = pred.get_output_handle(pred.get_output_names()[0]).copy_to_cpu()[0].argmax(-1)
    keep = np.r_[True, idx[1:] != idx[:-1]] & (idx != 0)
    return "".join(chars[i] for i in idx[keep])


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--save-preds", action="store_true", help="폴더에 ppocr_preds.json 저장")
    args = ap.parse_args()
    pred, chars = load()
    for d in map(Path, args.dirs):
        gt = json.loads((d / "transcriptions.json").read_text())
        e = n = 0; preds = {}; per = []
        for k, t in gt.items():
            p = preds[k] = read(pred, chars, d / k)
            p, t = norm(p), norm(t)
            ek = edit(p, t); e += ek; n += len(t); per.append(ek / max(1, len(t)))
        if args.save_preds:
            (d / "ppocr_preds.json").write_text(json.dumps(preds, ensure_ascii=False, indent=0))
        per = np.array(per)
        print(f"{d}\tCER_ppocr corpus {e / n:.4f}  평균 {per.mean():.4f}  중앙값 {np.median(per):.4f}  "
              f"폭주율 {(per > 1).mean():.1%}\t(n={len(gt)})", flush=True)


if __name__ == "__main__":
    main()
