"""get_model_inputs 가 style/gen 을 몇 % 쓰는지 측정 (STYLE_LEN_BUG 검증).
패치 전: style ~10~20% (배치최대/8 로 잘림).  패치 후(*8): ~100%.
"""
import torch
from models.eruku import Emuru

m = Emuru(); m.eval()

def used_tokens(mi, i):
    sp = mi["specials"][i]
    sog = (sp == 0).nonzero()                       # 첫 SOG 이전 = style Img 토큰 수
    style = sog[0].item() if len(sog) > 0 else (sp == 2).sum().item()
    eog = (sp == 1).nonzero()
    gen = (eog[0].item() - sog[0].item() - 1) if (len(sog) and len(eog)) else 0
    return style, gen

widths = [420, 560, 700, 380]                        # 서로 다른 짧은 라인들 (포크 실제 경로: list)
style = [torch.rand(3, 64, w) for w in widths]
gen = [torch.rand(3, 64, w) for w in widths]
mi = m.get_model_inputs(style, gen, style_len=widths, gen_len=widths, max_img_len=8192)

print(f"배치 폭={widths}  배치최대={max(widths)}px  latent_width={max(widths)//8}")
print(f"{'폭(px)':>7} | {'style 사용':>16} | {'gen 사용':>16}")
for i, w in enumerate(widths):
    st, gn = used_tokens(mi, i)
    print(f"{w:7d} | {st*8:5d}px ({st*8/w:4.0%}) | {gn*8:5d}px ({gn*8/w:4.0%})")
