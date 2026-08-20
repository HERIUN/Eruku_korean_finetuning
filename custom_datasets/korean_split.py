"""repo OnlineSplitFontSquare 를 한글로 사용 (온라인 style/gen split 렌더).

- style/gen 텍스트를 각각 다른 단어수 범위로 샘플 (Phase1: 둘 다 2~4 / Phase2: style 1~8, gen 1~32)
- 한글 mixed-script sampler (custom_datasets.korean_fontset.MixedLineSampler 재사용, 전역 covered=폰트 union)
- 폰트별 charset 필터(fonts_charsets.json)로 tofu 방지
- 무한 온라인 (디스크 0). 학습 샘플 저장은 dump_samples() 사용.

CLI 테스트:
  python custom_datasets/korean_split.py --n 8 --style 1 8 --gen 2 16 --out /tmp/ksplit
"""
from __future__ import annotations
import sys, random, json, argparse
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]   # 저장소 루트
ASSETS = HERE / "assets"
sys.path.insert(0, str(HERE))
import custom_datasets.korean_fontset as G
from custom_datasets.font_square.font_square_split import OnlineSplitFontSquare
from custom_datasets.font_square import font_transforms_split as FT
from custom_datasets.font_square import font_transforms as FTN  # 단일 텍스트(비-split) 증강
from torchvision import transforms as T


class KoreanSplitFontSquare(OnlineSplitFontSquare):
    """OnlineSplitFontSquare 와 동일하되 style/gen 을 서로 다른 sampler 로 뽑음.

    ★ 경계 버그 수정 + style/gen 정책(2026-06): 원본은 [style|gap|gen] 을 한 이미지로 합쳐 렌더 후
    RandomRotation·RandomWarping(비균일)을 전체 적용하고 고정 컬럼에서 잘라 → 변형이 경계를 밀어
    style/gen 이 서로 침범(잘림/잔상)했다. 수정: style/gen 을 **각각 따로 렌더(경계 없음)** + 단계별
    transform 명시 호출로 일부는 공유, 일부는 비대칭:
    - **공유(style=gen, RNG 스냅샷으로 동일 실현)**: 같은 폰트(writer), GaussianBlur(선명도),
      GrayscaleDilation(굵기), SplitAlphaChannel alpha_ink(잉크 투명도/희미함),
      ColorJitter(contrast·saturation·hue; **brightness=0**) → 잉크 질감/대비 매칭.
    - **비대칭**: RandomRotation(약한 기울기) = **style 만**(gen 곧은 글자) / 배경 = **gen 무조건 흰색**
      (style 종이배경). RandomWarping(TPS 휨) 은 **완전 제거**(style·gen 모두). ColorJitter brightness=0
      이라 흰배경은 거의 안 틀어짐(+0.99~1.0 유지).
    근거: 모델 generate() 는 흰배경·곧은 글자를 출력 → 타겟(gen)을 그렇게 학습해야 휨/배경 전파 안 됨.
    style 은 추론 시 실제 배경·필기 이미지라 현실적 증강 유지. 잉크 질감은 같은 writer 처럼 보이게 매칭.
    """
    def __init__(self, fonts, backgrounds, style_sampler, gen_sampler, legacy=False, **kw):
        super().__init__(fonts, backgrounds, text_sampler=style_sampler, **kw)
        self.style_sampler = style_sampler
        self.gen_sampler = gen_sampler
        # legacy=True: 원본 composite([style|gap|gen] 합쳐 렌더+전체증강+경계분할) 방식 사용(부모 self.transform).
        # 데이터 transform 변경 전/후 비교 실험용. RandomInvert 는 곱셈합성에서 무정보 → 원본처럼 비활성.
        self.legacy = legacy
        if legacy:
            for t in self.transform.transforms:
                if isinstance(t, FT.RandomInvert):
                    t.p = 0.0
        renderers = self.transform.transforms[0].renderers
        bg_dir = Path(backgrounds)
        bg_paths = ([p for p in bg_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
                    if bg_dir.is_dir() else list(backgrounds))
        self.render_tf = FTN.RenderImage(self.fonts, renderers=renderers, pad=20)
        # 단계별 transform (Compose 대신 명시 호출): 공유 단계(블러·굵기)는 RNG 스냅샷으로 둘에 동일 적용.
        self.t_rotate = FTN.RandomRotation(3, fill=1, p=0.5)           # style 만(약한 기울기). warp 는 제거됨.
        self.t_blur = FTN.GaussianBlur(kernel_size=3, p=0.5)          # ★ 공유(선명도 매칭)
        self.t_bg = FTN.RandomBackground(bg_paths, white_p=0.1)        # style 만(종이배경)
        self.t_tailor = FTN.TailorTensor(pad=3)
        self.t_split = FTN.SplitAlphaChannel()                        # ★ 공유(alpha_ink 0.5~1.0 매칭)
        self.t_dilate = FTN.GrayscaleDilation(kernel_size=2, p=0.1)   # ★ 공유(굵기 매칭)
        # ★ 공유: ColorJitter — brightness=0(끔), contrast·saturation·hue 만 style/gen 동일 적용
        self.t_jitter = FTN.ColorJitter(brightness=0, contrast=0.05, saturation=0.05, hue=0.05, p=0.5)
        self.t_resize = FTN.ImgResize(64)
        self.t_merge = FTN.MergeWithBackground()
        self.t_norm = FTN.Normalize((0.5,), (0.5,))

    @staticmethod
    def _snap():
        return random.getstate(), np.random.get_state(), torch.get_rng_state()

    @staticmethod
    def _restore(st):
        random.setstate(st[0]); np.random.set_state(st[1]); torch.set_rng_state(st[2])

    def __getitem__(self, font_id):
        font_id = font_id % self.renderers_length
        style_text, gen_text = self.style_sampler(), self.gen_sampler()
        # ── legacy: 원본 composite 방식 (합쳐 렌더 → 전체 증강 → 고정컬럼 분할) ──
        if self.legacy:
            sample = self.transform({'style_text': style_text, 'gen_text': gen_text, 'font_id': font_id})
            sw = sample['style_img_width'] * sample['img'].shape[2] // sample['total_img_width']
            return {'style_img': sample['img'][:, :, :sw], 'gen_img': sample['img'][:, :, sw:],
                    'style_text': sample['style_text'], 'gen_text': sample['gen_text'], 'writer': font_id}
        # ── 신 방식: style/gen 독립 렌더(경계 없음), 같은 폰트(=같은 writer) ──
        s = self.render_tf({'text': style_text, 'font_id': font_id})
        g = self.render_tf({'text': gen_text, 'font_id': font_id})

        # style 만: 약한 기울기(회전). warp(TPS 휨) 는 제거. gen 은 곧은 글자.
        s = self.t_rotate(s)

        # ★ 공유: GaussianBlur — 같은 결정/시그마를 style·gen 에 (선명도 매칭)
        st = self._snap(); s = self.t_blur(s); self._restore(st); g = self.t_blur(g)

        # 배경: style = 종이배경(증강), gen = 흰배경 강제
        s = self.t_bg(s)
        _, h, w = g['img'].shape
        g['bg_patch'] = torch.ones((3, h, w))

        # 둘 다: 크롭
        s = self.t_tailor(s); g = self.t_tailor(g)
        # ★ 공유: SplitAlphaChannel — 같은 alpha_ink(잉크 투명도/희미함)를 둘에
        st = self._snap(); s = self.t_split(s); self._restore(st); g = self.t_split(g)
        # ★ 공유: GrayscaleDilation — 같은 굵기 결정을 둘에
        st = self._snap(); s = self.t_dilate(s); self._restore(st); g = self.t_dilate(g)

        # ★ 공유: ColorJitter(brightness=0, contrast·sat·hue) — 같은 실현을 style·gen 에
        st = self._snap(); s = self.t_jitter(s); self._restore(st); g = self.t_jitter(g)

        # 둘 다: 리사이즈 → 배경합성 → 정규화
        for t in (self.t_resize, self.t_merge, self.t_norm):
            s = t(s); g = t(g)
        return {
            'style_img': s['img'],
            'gen_img': g['img'],
            'style_text': s['text'],
            'gen_text': g['text'],
            'writer': font_id,
        }


def build_samplers(style_range, gen_range, n_english=8000, seed=42):
    rng = random.Random(seed)
    pools = G.build_pools(ASSETS / "corpus/korean_lines.txt",
                          ASSETS / "corpus/chars.txt",
                          str(ASSETS / "corpus/english_words.txt"), n_english, rng)
    cs = json.load(open(ASSETS / "fonts_korean_v2/train/fonts_charsets.json"))
    inter = None
    for s in cs.values():
        f = {ord(c) for c in s}
        inter = f if inter is None else inter & f
    # covered_cps = 전 폰트 charset 의 **교집합**. 합집합을 쓰면 "어떤 폰트든 하나라도 그리면 통과"라
    # 정작 배정된 폰트가 못 그리는 글자(…“”‘’~)가 통과했다가 렌더 직전에 조용히 삭제된다
    # (라인의 7.1%). 교집합으로 좁혀도 ko/en 풀은 100% 살아남아 잃는 게 없다. docs/DATA_LIMITATIONS.md D8.
    cps = inter
    # rand 음절 풀 = 그 교집합의 한글 음절 (KS X 1001 2350자) → tofu-safe
    syls = sorted(chr(c) for c in inter if 0xAC00 <= c <= 0xD7A3)
    w = {"ko": 0.45, "en": 0.20, "num": 0.20, "rand": 0.15}
    # punct/wrap 은 독립 확률. 코퍼스 실측(후행 7.6% / 감싸기 0.1%) 대비 글리프 노출을 위해
    # 여전히 상향이지만, 구 값(0.35 / 0.14=0.35*0.4 매직넘버)의 과다 노출은 줄였다. D9.
    punct, wrap, special = 0.20, 0.05, 0.08
    style_s = G.MixedLineSampler(pools["ko"], pools["en"], cps, w, style_range[0], style_range[1], 40,
                                 punct, special, rng, rand_syllables=syls, wrap_prob=wrap)
    gen_s = G.MixedLineSampler(pools["ko"], pools["en"], cps, w, gen_range[0], gen_range[1], 130,
                               punct, special, rng, rand_syllables=syls, wrap_prob=wrap)
    print(f"samplers: style {style_range} gen {gen_range} | ko {len(pools['ko'])} en {len(pools['en'])} "
          f"rand_syl {len(syls)} cps {len(cps)} (font charset 교집합) "
          f"| punct {punct} wrap {wrap} special {special}")
    return style_s, gen_s


def split_train_fonts(val_frac, seed=42, fonts_dir=None):
    """학습 폰트를 (train_fonts, val_fonts) 리스트로 결정적 분할. val_frac = val 폰트 비율.

    같은 (val_frac, seed) 면 항상 같은 분할 → run 간 재현 가능. val_frac=0 이면 val 은 빈 리스트."""
    d = Path(fonts_dir) if fonts_dir else (ASSETS / "fonts_korean_v2/train")
    fonts = sorted(p for p in d.glob("*") if p.suffix.lower() in (".ttf", ".otf"))
    idx = list(range(len(fonts)))
    random.Random(seed).shuffle(idx)
    k = max(1, int(round(len(fonts) * val_frac))) if val_frac > 0 else 0
    val_i = set(idx[:k])
    return ([f for i, f in enumerate(fonts) if i not in val_i],
            [f for i, f in enumerate(fonts) if i in val_i])


def make_dataset(style_range=(1, 8), gen_range=(1, 32), length=None, seed=42, fonts_dir=None, legacy=False):
    """fonts_dir: None=학습 폰트 전체(fonts_korean_v2/train) / 디렉토리 경로 / 폰트 경로 리스트
    (split_train_fonts 의 분할 결과). held-out 은 fonts_korean_v2/test 디렉토리 전달.
    legacy=True: 원본 composite transform(데이터 정책 변경 전) 사용."""
    style_s, gen_s = build_samplers(style_range, gen_range, seed=seed)
    if fonts_dir is None:
        fonts_dir = ASSETS / "fonts_korean_v2/train"
    elif not isinstance(fonts_dir, (list, tuple)):
        fonts_dir = Path(fonts_dir)
    return KoreanSplitFontSquare(fonts_dir,
                                 str(ASSETS / "backgrounds"),
                                 style_s, gen_s, length=length, legacy=legacy)


def build_english_samplers(style_range, gen_range, fonts_dir, n_english=8000, seed=42):
    """영어 전용(라틴 폰트) sampler. ko=0, en+num 만 → 한글 토푸 방지.
    cps = 영어 폰트 charset union (없으면 ASCII 가시문자).

    ★ 한글 경로(build_samplers)는 교집합을 쓰지만 여기는 **union 유지**다. 영어 폰트 177종의
    교집합은 63자(A-Za-z0-9+공백)뿐이라 구두점이 전멸하고 gen_number 의 `%,-.:` 까지 걸려
    숫자가 randint 폴백만 남는다. 대신 렌더 단계의 글자 삭제를 그대로 감수한다.
    docs/DATA_LIMITATIONS.md D8 참고."""
    rng = random.Random(seed)
    pools = G.build_pools(ASSETS / "corpus/korean_lines.txt", ASSETS / "corpus/chars.txt",
                          str(ASSETS / "corpus/english_words.txt"), n_english, rng)
    csf = Path(fonts_dir) / "fonts_charsets.json"
    cps = set()
    if csf.exists():
        for s in json.load(open(csf)).values():
            cps |= {ord(c) for c in s}
    else:
        cps = set(range(0x20, 0x7f))
    w = {"ko": 0.0, "en": 0.80, "num": 0.20, "rand": 0.0}   # 영어 단어 + 숫자만
    punct, wrap, special = 0.20, 0.05, 0.08                 # 한글 경로와 동일 (D9)
    style_s = G.MixedLineSampler(pools["ko"], pools["en"], cps, w, style_range[0], style_range[1],
                                 40, punct, special, rng, rand_syllables=[], wrap_prob=wrap)
    gen_s = G.MixedLineSampler(pools["ko"], pools["en"], cps, w, gen_range[0], gen_range[1],
                               130, punct, special, rng, rand_syllables=[], wrap_prob=wrap)
    print(f"english samplers: style {style_range} gen {gen_range} | en {len(pools['en'])} cps {len(cps)} "
          f"(union) | punct {punct} wrap {wrap} special {special}")
    return style_s, gen_s


def make_english_dataset(style_range=(1, 8), gen_range=(1, 32), length=None, seed=42, fonts_dir=None):
    """라틴 폰트로 영어 전용 split 데이터. 학습에 한글 데이터와 섞어 영어 스타일 다양성 보강."""
    assert fonts_dir, "english fonts_dir 필요"
    style_s, gen_s = build_english_samplers(style_range, gen_range, fonts_dir, seed=seed)
    return KoreanSplitFontSquare(Path(fonts_dir), str(ASSETS / "backgrounds"),
                                 style_s, gen_s, length=length)


def split_collate(batch):
    """split 데이터셋(style_img/gen_img/...) → train.core 형식(style_img/gen_img + len + text)."""
    return {
        "style_img": [b["style_img"] for b in batch],
        "gen_img": [b["gen_img"] for b in batch],
        "style_text": [b["style_text"] for b in batch],
        "gen_text": [b["gen_text"] for b in batch],
        "style_img_len": [b["style_img"].shape[-1] for b in batch],
        "gen_img_len": [b["gen_img"].shape[-1] for b in batch],
    }


def to_gray(t):  # [3,H,W] in [-1,1] -> uint8 grayscale
    return (((t.mean(0) + 1) / 2).clamp(0, 1).numpy() * 255).astype(np.uint8)


def model_style_recon(model, style_img, gen_img, max_img_len):
    """'모델이 실제로 받는 style' 을 복원 (학습 입력 = 데이터 파이프라인 검증용).

    학습 루프와 **동일한** `model.get_model_inputs` 를 그대로 호출한다 → style-len 단위fix
    (픽셀↔latent, docs/STYLE_LEN_BUG.md)가 반영된 실제 입력이 나온다. 반환 embeds 의 style
    구간(첫 SOG 이전)만 VAE 디코딩해 이미지로 복원. 만약 fix 가 되돌려지면 여기서 style 이
    1/8 로 잘려 그대로 눈에 보인다(= 회귀 검증). budget(max_img_len//2) 초과 style 도 잘림이 보인다.

    반환: (복원 grayscale [64,W'], 복원폭 px, 원본 style 폭 px)
    """
    from einops import rearrange
    with torch.no_grad():
        mi = model.get_model_inputs([style_img], [gen_img],
                                    int(style_img.shape[-1]), int(gen_img.shape[-1]), max_img_len)
        emb = mi["decoder_inputs_embeds"][0]     # [seq, 8]
        sp = mi["specials"][0]                    # [seq]  (2=Img, 0=SOG, 1=EOG)
        sog = (sp == 0).nonzero().flatten()
        end = int(sog[0].item()) if len(sog) else int((sp == 2).sum().item())
        end = max(1, end)
        lat = rearrange(emb[:end], 'w (c h) -> 1 c h w', c=1, h=8).to(model.vae.device)
        img = model.vae.decode(lat).sample        # [1,C,64,end*8] ~[-1,1]
    im = ((img.clamp(-1, 1) + 1) / 2)[0]
    if im.shape[0] > 1:
        im = im.mean(0, keepdim=True)
    arr = (im[0].cpu().numpy() * 255).astype(np.uint8)
    return arr, end * 8, int(style_img.shape[-1])


def dump_samples(ds, n, out_dir, model=None, max_img_len=2048):
    """학습에 쓰이는 형식의 샘플 n개를 저장 + montage.

    model 지정 시(=Emuru): 각 행에 '모델이 실제로 받는 style(복원)' 열을 추가해, 데이터
    파이프라인에서 style-len 단위fix 가 반영됐는지 end-to-end 로 확인 (model_style_recon).
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    gf = ImageFont.truetype(str(ASSETS / "fonts_label/NanumGothic-Regular.ttf"), 15)
    rows = []
    for i in range(n):
        s = ds[i]
        si, gi = to_gray(s["style_img"]), to_gray(s["gen_img"])
        cv2.imwrite(str(out / f"{i:03d}_style.png"), si)
        cv2.imwrite(str(out / f"{i:03d}_gen.png"), gi)
        # montage row: label + [style | sep | gen  (| sep | 모델입력 style)]
        H = 64
        def fit(a):
            if a.shape[0] != H:
                a = cv2.resize(a, (max(1, int(a.shape[1] * H / a.shape[0])), H))
            return a
        si, gi = fit(si), fit(gi)
        sep = np.full((H, 6), 0, np.uint8)
        parts = [si, sep, gi]
        cov_txt = ""
        if model is not None:
            recon, used_px, orig_px = model_style_recon(model, s["style_img"], s["gen_img"], max_img_len)
            cv2.imwrite(str(out / f"{i:03d}_model_style.png"), recon)
            parts += [np.full((H, 6), 0, np.uint8), fit(recon)]
            cov = used_px / max(1, orig_px)
            cov_txt = f" | [model-style {used_px}/{orig_px}px {cov:.0%}]"
        body = np.hstack(parts)
        lab = Image.new("L", (body.shape[1], 22), 235)
        ImageDraw.Draw(lab).text((4, 3), f"[style:{s['style_text']}] | [gen:{s['gen_text']}]{cov_txt}", font=gf, fill=0)
        rows.append(np.vstack([np.array(lab), body, np.full((4, body.shape[1]), 90, np.uint8)]))
        print(f"  {i}: style={s['style_text']!r} ({si.shape[1]}px) | gen={s['gen_text']!r} ({gi.shape[1]}px)"
              + (cov_txt if model is not None else ""))
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1]), 255, np.uint8)]) if r.shape[1] < W else r for r in rows]
    Image.fromarray(np.vstack(rows)).save(out / "_montage.png")
    print(f"saved {out}/_montage.png + {n} style/gen pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--style", nargs=2, type=int, default=[1, 8])
    ap.add_argument("--gen", nargs=2, type=int, default=[2, 16])
    ap.add_argument("--out", default="/tmp/ksplit")
    ap.add_argument("--show-model-input", action="store_true",
                    help="각 행에 '모델이 실제로 받는 style(복원)' 열 추가 → style-len 단위fix 가 "
                         "데이터 파이프라인에 반영됐는지 검증. Emuru(VAE+get_model_inputs) 로드 필요.")
    ap.add_argument("--max-img-len", type=int, default=2048,
                    help="--show-model-input 시 get_model_inputs 예산 (학습값과 맞추면 실제 잘림 재현).")
    args = ap.parse_args()
    ds = make_dataset(style_range=tuple(args.style), gen_range=tuple(args.gen), length=10000)
    model = None
    if args.show_model_input:
        # T5 는 config-init(랜덤)이라 가중치 다운로드 없음. 필요한 건 pretrained VAE + get_model_inputs
        # 슬라이스 로직뿐(학습 가중치 무관 = 어떤 ckpt 든 style 잘림 결과 동일).
        from models.eruku import Emuru
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("loading Emuru (VAE + get_model_inputs) for model-input view ...")
        model = Emuru(t5_checkpoint="google-t5/t5-large",
                      vae_checkpoint="blowing-up-groundhogs/emuru_vae").to(dev).eval()
    dump_samples(ds, args.n, args.out, model=model, max_img_len=args.max_img_len)
