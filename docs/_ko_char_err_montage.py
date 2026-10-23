"""한글 decode A/B 에서 '정답 렌더는 PP-OCR 이 맞게 읽는데 생성에서 틀린' 글자의 줄 몽타주.
입력: finetune_runs/_ppocr_ko/decode_ab/<Ts|noTs>/<gt|all|ctx>/ (ppocr_rec.py --save-preds 결과)
출력: docs/img_decode/ko_char_errors.png  (빨간 박스 = 틀린 글자의 대략 위치: 글자폭 가중 근사)"""
import json, re
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R = Path(__file__).resolve().parents[1]
D = R / "finetune_runs/_ppocr_ko/decode_ab"
FONT = ImageFont.truetype(str(R / "assets/fonts_korean_v2/train/GothicA1.ttf"), 15)
H, W, LW = 44, 1300, 70
norm = lambda s: re.sub(r"\s+", " ", s).strip()

def align(t, p):
    n, m = len(t), len(p); d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (t[i-1] != p[j-1]))
    out, i, j = [None] * n, n, m
    while i > 0:
        if j > 0 and d[i][j] == d[i-1][j-1] + (t[i-1] != p[j-1]): out[i-1] = p[j-1]; i -= 1; j -= 1
        elif d[i][j] == d[i-1][j] + 1: out[i-1] = "∅"; i -= 1
        else: j -= 1
    return out

def errors(cond, mode):
    """[(key, 위치, 목표글자, 읽힌글자)] — 정답 렌더 정독 위치만."""
    tr = json.load(open(D / cond / "gt/transcriptions.json"))
    gt = json.load(open(D / cond / "gt/ppocr_preds.json"))
    pr = json.load(open(D / cond / mode / "ppocr_preds.json"))
    out = []
    for k, t in tr.items():
        t = norm(t)
        for i, (c, g, x) in enumerate(zip(t, align(t, norm(gt[k])), align(t, norm(pr[k])))):
            if c != " " and g == c and x != c: out.append((k, i, c, x))
    return out

SEEN = set()
def pick(errs, n):
    """치환 유형((목표,읽힘))이 겹치지 않게, 흔한 유형부터 n 개 (이미 쓴 줄은 건너뜀)."""
    by = defaultdict(list)
    for e in errs: by[(e[2], e[3])].append(e)
    out = []
    for _, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        e = next((e for e in v if e[0] not in SEEN), None)
        if e and len(out) < n: out.append(e); SEEN.add(e[0])
    return out

def cw(ch):   # 렌더 글자폭 근사 (한글 1, 공백·라틴·구두점은 좁다)
    return 1.0 if "\uac00" <= ch <= "\ud7a3" else 0.35 if ch == " " else 0.55

def load(p):
    im = Image.open(p).convert("RGB"); w = max(1, im.width * H // im.height)
    return im.resize((w, H)).crop((0, 0, min(w, W - LW), H))

picks = [("Ts", e) for e in pick(errors("Ts", "ctx"), 7)] + \
        [("noTs", e) for e in pick(errors("noTs", "ctx"), 4)] + \
        [("noTs", e) for e in pick([e for e in errors("noTs", "all")], 4)]
tr = {c: json.load(open(D / c / "gt/transcriptions.json")) for c in ("Ts", "noTs")}
preds = {(c, m): json.load(open(D / c / m / "ppocr_preds.json")) for c in ("Ts", "noTs") for m in ("gt", "all", "ctx")}
blocks = []
for cond, (k, i, c, x) in picks:
    t = norm(tr[cond][k])
    rows = []
    for m in ("gt", "all", "ctx"):
        im = load(D / cond / m / k)
        dr = ImageDraw.Draw(im)
        cx = int(im.width * (sum(map(cw, t[:i])) + cw(t[i]) / 2) / sum(map(cw, t))) if m == "gt" else None   # 글자폭 가중 위치 근사
        if cx: dr.rectangle((cx - H * 2 // 5, 0, cx + H * 2 // 5, H - 1), outline=(220, 30, 30), width=2)
        rows.append((m, im, norm(preds[(cond, m)][k])))
    blocks.append((f"[{'w/ T_s' if cond == 'Ts' else 'w/o T_s'}] '{c}' → '{x}'   정답: {t}", rows))
hdr, rh = 22, H + 22
canvas = Image.new("RGB", (W, sum(hdr + 3 * rh + 8 for _ in blocks)), "white")
dr, y = ImageDraw.Draw(canvas), 0
for title, rows in blocks:
    dr.text((4, y + 3), title, font=FONT, fill=(0, 0, 0)); y += hdr
    for m, im, p in rows:
        dr.text((6, y + H // 2 - 8), {"gt": "정답렌더", "all": "all", "ctx": "ctx"}[m], font=FONT, fill=(90, 90, 90))
        canvas.paste(im, (LW, y))
        dr.text((LW, y + H + 2), f"PP-OCR: {p}", font=FONT, fill=(30, 90, 200)); y += rh
    dr.line((0, y + 4, W, y + 4), fill=(170, 170, 170)); y += 8
out = R / "docs/img_decode/ko_char_errors.png"; canvas.save(out); print(out, canvas.size)
