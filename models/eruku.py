
import torch
import os as _os
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config
from custom_datasets import HFDataCollector
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch.nn import MSELoss, CTCLoss, CrossEntropyLoss
from pathlib import Path
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
from models.origami import OrigamiNet
from diffusers import AutoencoderKL
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn as nn
from typing import Tuple

# Safer defaults for clearer NCCL error reporting during debugging
# ★ CUDA_LAUNCH_BLOCKING 은 여기서 강제하지 않는다: 커널 런치를 동기화해 CUDA 에러를
#   진짜 발생 지점에 붙여주지만, 자기회귀 생성처럼 작은 커널이 많은 경로가 1.5~1.8배
#   느려진다(실측: 배치 생성 8샘플 2.4s → 4.4s). 디버깅할 때만 밖에서 켤 것 —
#   `CUDA_LAUNCH_BLOCKING=1 python ...`
_os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _len_at(value, idx: int) -> int:
    """샘플 idx 의 길이(픽셀)를 int 로. int(전 샘플 공용) / list / 텐서 모두 받는다.

    ★ 조용한 fallback 을 두지 않는다: 여기서 길이를 잘못 집으면 style 이 잘린 채로
      학습·생성이 멀쩡히 돌아가 버린다(docs/STYLE_LEN_BUG.md 가 딱 그 사고였다).
      변환이 실패하면 그대로 터뜨려서 호출부를 고치게 한다.
    """
    return int(value if isinstance(value, int) else value[idx])


def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    padded = rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')
    return padded.contiguous()




# sog, eog, img
SPECIAL_TOKEN_COUNT = 3


# ─────────────────────────────────────────────────────────────────────────────
# C1: 자모 분해 텍스트 조건 (docs/JONGSEONG_STROKE_LOSS.md)
# ─────────────────────────────────────────────────────────────────────────────
# 텍스트 조건은 byT5 = UTF-8 **바이트** 다. 한글 완성형(3바이트)은 종성 구조를 바이트로
# 드러내지 않는다 — 같은 ㄺ 받침을 쓰는 닭 eb8bad / 읽 ec9dbd / 맑 eba791 이 바이트를
# 하나도 공유하지 않는다(코드 = 0xAC00 + 초성·588 + 중성·28 + 종성 인데 28 이 바이트 경계
# 64 와 안 맞아 종성이 비선형으로 흩어짐). 그래서 모델은 "ㄺ 은 이렇게 그린다" 를 일반화하지
# 못하고 음절을 통째로 외운다(실측: 음절단위 노출 β +0.152 vs 종성단위 노출 β −0.040).
class Emuru(torch.nn.Module):
    def __init__(self, t5_checkpoint='google-t5/t5-base',
                 vae_checkpoint='blowing-up-groundhogs/emuru_vae',
                 ocr_checkpoint=None, slices_per_query=1, channels=1, text_dropout_probability=0.0, img_dropout_probability=0.0):
        super(Emuru, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')  # per-character tokenizer
        self.tokenizer.add_tokens(["<sog>"])
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)
        self.t5_name_or_path = t5_checkpoint

        config = T5Config.from_pretrained(t5_checkpoint)
        config.vocab_size = len(self.tokenizer)
        self.T5 = T5ForConditionalGeneration(config)
        # Expose a HF-like config for downstream trainers expecting model.config
        self.config = self.T5.config
        # Ensure a valid identifier is present for downstream AutoProcessor lookups
        try:
            if not getattr(self.config, "_name_or_path", None):
                self.config._name_or_path = str(self.t5_name_or_path)
        except Exception:
            # As a safe fallback, set attribute directly
            self.config._name_or_path = str(self.t5_name_or_path)
        self.T5.lm_head = torch.nn.Identity()
        self.sos = torch.nn.Embedding(1, config.d_model)
        self.sog = torch.nn.Embedding(1, config.d_model)
        self.eog = torch.nn.Embedding(1, config.d_model)
        
        self.vae = AutoencoderKL.from_pretrained(vae_checkpoint)
        
        vae_latent_dim = 8 # self.vae.config.get('latent_channels', 8)

        self.query_emb = torch.nn.Linear(vae_latent_dim * channels * slices_per_query, config.d_model)
        self.t5_to_vae = torch.nn.Linear(config.d_model, vae_latent_dim * channels * slices_per_query)
        self.t5_to_special = torch.nn.Linear(config.d_model, SPECIAL_TOKEN_COUNT)
        self.t5_to_ocr = torch.nn.Linear(config.d_model, len(self.tokenizer), bias=False)

        self.uncond_embedding = torch.nn.Embedding(1, config.d_model)
        self.dropout_probability = 0.0
        self.drop_text = False
        self.drop_img = False

        self.set_training(self.vae, False)

        # OCR(OrigamiNet, 영어) 은 alpha<1.0 일 때만 loss 에 쓰임. 한글 학습은 alpha=1.0
        # 이라 불필요 — 공식 HF 릴리즈(modeling_eruku.py)에도 OCR 모듈이 없음.
        if ocr_checkpoint is not None:
            self.ocr = OrigamiNet.from_checkpoint(ocr_checkpoint, o_classes=165, n_channels=1)
            self.set_training(self.ocr, False)
        else:
            self.ocr = None
        
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=channels, q=slices_per_query)

        self.mse_criterion = MSELoss()#(reduction='none') # TODO:change reductions if you intend to add a mask
        self.ce_criterion = CrossEntropyLoss()
        # self.ctc_criterion = CTCLoss()
        self.trainer = None
        self.alpha = 1.0
        # Minimal attributes for TRL compatibility
        self.warnings_issued = {}
        self._model_tags = set()

    def add_model_tags(self, tags):
        try:
            if isinstance(tags, (list, tuple, set)):
                self._model_tags.update(tags)
            elif isinstance(tags, str):
                self._model_tags.add(tags)
        except Exception:
            # No-op if tags updating fails
            pass

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing - delegate to T5 model"""
        if hasattr(self.T5, 'gradient_checkpointing_enable'):
            self.T5.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing - delegate to T5 model"""
        if hasattr(self.T5, 'gradient_checkpointing_disable'):
            self.T5.gradient_checkpointing_disable()

    def set_training(self, model, training):
        model.train() if training else model.eval()
        for param in model.parameters():
            param.requires_grad = training 
    
    def _img_encode(self,img):
        # ★ 이중정규화 제거: 입력 img 는 이미 [-1,1] (dataset t_norm / infer style_tensor 모두
        #   Normalize(0.5,0.5) 적용) = VAE 표준 입력범위. upstream 원본은 여기서 Normalize(0.5,0.5) 를
        #   한 번 더 적용해 [-3,1] 로 밀어 VAE 재구성이 ~3.4배 악화됐다(docs/STYLE_LEN_BUG.md).
        #   실측: single(정상[-1,1]) 로 바꿔도 생성 붕괴 없음(MSE 소폭 개선) → 제거.
        # Ensure contiguous memory layout before encode to avoid kernel issues
        img = img.contiguous()
        return self.vae.encode(img.float()).latent_dist.sample()

    @torch.no_grad()
    def get_model_inputs(self, style_img, gen_img, style_len, gen_len, max_img_len):
        bs = len(style_img)
        decoder_inputs_embeds_list = []
        specials_list = []
        lengths_list = []
        
        # Move images to device and pad them
        style_img = pad_images([el.to(self.T5.device) for el in style_img])
        
        if gen_img is not None:
            gen_img = pad_images([el.to(self.T5.device) for el in gen_img])
            gen_img_embeds = self._img_encode(gen_img)
        else:
            gen_img_embeds = None

        style_img_embeds = self._img_encode(style_img)
        
        for el in range(bs):
            sl = _len_at(style_len, el)

            # Ensure widths are within bounds.
            # 원본 validate_widths 와 동일: style 은 max_img_len 의 절반까지만 차지하게 제한.
            # ※ 학습에서는 이 clamp 가 발동하지 않는다 — train.core 가 filter_no_truncation 으로
            #   예산 초과 샘플을 **미리 버리기** 때문(no-truncation 정책, e02e1d9). 따라서 절반 캡은
            #   "긴 style 을 학습에 쓸 것인가"라는 정책 선택이지, 잘림을 막는 장치가 아니다.
            #   없애면 판정이 합 조건 하나(style + gen + 16 <= max_img_len)로 단순해진다.
            #   실제로 이 캡 때문에만 버려지는 샘플은 전체의 0.016%(75/480,000) 였다.
            # ※ 반면 **추론에는 drop 이 없어** 여기서 진짜로 잘린다. style 이 max_img_len/2 를
            #   넘으면 뒷부분이 조건에서 빠진다 → 추론 max_img_len 을 학습과 맞출 것
            #   (configs/{eval,infer}.yaml 의 max_img_len).
            # ★ 단위주의: sl 은 '픽셀', style_img_embeds.shape[-1] 은 'latent 폭'(=픽셀/8).
            #   원본은 style 을 고정 8192px 로 패딩(pad_images_fixed)해 latent 폭을 크게 만들어
            #   이 항이 발동 안 하게 했지만, 여기선 배치최대 폭으로만 패딩하므로 *8 로 픽셀 환산해야
            #   style 이 1/8 로 잘리지 않는다. (docs/STYLE_LEN_BUG.md 참고)
            sl = max(64, min(sl, style_img_embeds.shape[-1] * 8, max_img_len // 2))

            # Start with style image embeds
            sample_embeds_parts = [style_img_embeds[el,:,:,:sl//8]]
            specials_parts = [torch.ones(sl//8) * 2] # Img token

            if gen_img_embeds is not None and gen_len is not None:
                gl = _len_at(gen_len, el)

                # 원본 validate_widths 와 동일: gen 은 style 을 뺀 남은 budget 까지만.
                # (SOG+EOG 2 토큰 = 16px 예약) → 전개하면 style + gen + 16 <= max_img_len 과 같다.
                # 학습에서는 위와 같은 이유로 발동하지 않는다(초과 샘플은 이미 drop 됨).
                # ★ 단위주의: gen_img_embeds.shape[-1] 은 latent 폭 → *8 로 픽셀 환산 (위 sl 과 동일 이유)
                remaining = max(64, max_img_len - sl - 16)
                gl = max(64, min(gl, gen_img_embeds.shape[-1] * 8, remaining))
                sample_embeds_parts.extend([
                    torch.ones(1, 8, 1).to(self.T5.device), # SOG token placeholder
                    gen_img_embeds[el,:,:,:gl//8],
                    torch.ones(1, 8, 1).to(self.T5.device), # EOG token placeholder
                ])
                specials_parts.extend([
                    torch.zeros(1), # SOG
                    torch.ones(gl//8) * 2, # Img
                    torch.ones(1), # EOG
                ])

            sample_embeds = torch.cat(sample_embeds_parts, dim=-1)

            h_dim = sample_embeds.shape[1]
            sample_embeds = rearrange(sample_embeds, 'c h w -> w (h c)', h=h_dim, c=1)

            decoder_inputs_embeds_list.append(sample_embeds)
            lengths_list.append(sample_embeds.size(0))

            sample_specials = torch.cat(specials_parts, dim=0).to(self.T5.device)
            specials_list.append(sample_specials)

        # Pad sequences and ensure consistent shapes
        decoder_inputs_embeds_padded = pad_sequence(decoder_inputs_embeds_list, padding_value=1, batch_first=True)
        specials_padded = pad_sequence(specials_list, padding_value=1, batch_first=True)

        # Ensure we don't exceed max_img_len
        max_seq_len = max_img_len // 8
        if decoder_inputs_embeds_padded.size(1) > max_seq_len:
            decoder_inputs_embeds_padded = decoder_inputs_embeds_padded[:, :max_seq_len]
        if specials_padded.size(1) > max_seq_len:
            specials_padded = specials_padded[:, :max_seq_len]

        return {
            'decoder_inputs_embeds': decoder_inputs_embeds_padded,
            'specials': specials_padded.long(),
            # 샘플별 '실제' 길이(패딩 제외, latent 열 단위). 배치 생성에서
            # generate_batch(prefix_lens=...) 로 넘겨야 패딩을 style 로 먹지 않는다.
            'lengths': [min(el, max_seq_len) for el in lengths_list],
        }

    def _encode_text(self, style_text, gen_text):
        """style/gen 텍스트 → (input_ids, attention_mask). 학습·추론 공용 진입점."""
        pairs = [f"{style}<sog>{gen}" for style, gen in zip(style_text, gen_text)]
        enc = self.tokenizer(pairs, padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def forward(self, decoder_inputs_embeds_vae, specials, style_text, gen_text, ce_multiplier=1.0):
        # style_img_embeds: [bs, w//8, 8, 1]
        # generate text embeddings
        
        with torch.no_grad():
            text_ids, text_am = self._encode_text(style_text, gen_text)
        
        # add special tokens to img
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=decoder_inputs_embeds_vae.size(0))
        sog = repeat(self.sog.weight, '1 d -> b d', b=decoder_inputs_embeds_vae.size(0))
        # eog = repeat(self.eog.weight, '1 d -> b d', b=decoder_inputs_embeds_vae.size(0))

        decoder_inputs_embeds = self.query_emb(decoder_inputs_embeds_vae)
        
        # Fix the indexing assignment to avoid shape mismatch
        sog_mask = (specials == 0)
        eog_mask = (specials == 1)
        
        # Expand sog to match the sequence dimension
        sog_expanded = sog.unsqueeze(1).expand(-1, decoder_inputs_embeds.size(1), -1)
        
        if sog_mask.any():
            decoder_inputs_embeds[sog_mask] = sog_expanded[sog_mask]
        
        if eog_mask.any():
            # Expand eog to match the sequence dimension
            eog_expanded = self.eog.weight.unsqueeze(0).expand(decoder_inputs_embeds.size(0), decoder_inputs_embeds.size(1), -1)
            decoder_inputs_embeds[eog_mask] = eog_expanded[eog_mask]

        decoder_inputs_embeds = torch.cat(
            [
                sos,
                decoder_inputs_embeds
            ], dim = 1,
        )

        inputs_embeds = self.T5.shared(text_ids.to(self.T5.device))
        drop_ids = torch.rand(inputs_embeds.shape[0], device=inputs_embeds.device) < self.dropout_probability
        if self.drop_text:
            inputs_embeds = torch.where(drop_ids[:, None, None], self.uncond_embedding.weight, inputs_embeds)
        if self.drop_img:
            decoder_inputs_embeds = torch.where(drop_ids[:, None, None], self.uncond_embedding.weight, decoder_inputs_embeds)
        
        output = self.T5(inputs_embeds=inputs_embeds, attention_mask=text_am.to(self.T5.device),
                         decoder_inputs_embeds=decoder_inputs_embeds)
        
        vae_latent = self.t5_to_vae(output.logits[:, :-1])
        special_pred = self.t5_to_special(output.logits[:, :-1]) # [bs, w//8, 3]
        pred_latent = self.z_rearrange(vae_latent)

        ce_loss = ce_multiplier * self.ce_criterion(special_pred.flatten(0,1), specials.flatten(0,1))

        mse_mask = (specials == 2).unsqueeze(2) # [bs, w//8] TODO:consider putting the mask back in
        gt = decoder_inputs_embeds_vae * mse_mask
        vae_latent = vae_latent * mse_mask
        mse_loss = self.mse_criterion(vae_latent, gt)#/mse_mask.sum()
        ocr_loss = 0

        if self.alpha < 1.0:
            if self.ocr is None:
                raise RuntimeError("alpha<1.0 (OCR loss) 사용하려면 ocr_checkpoint 필요")
            pred_img = self.vae.decode(pred_latent).sample
            gt_img = self.vae.decode(decoder_inputs_embeds_vae.unsqueeze(1)).sample
            ocr_preds = self.ocr(pred_img)
            ocr_gt = self.ocr(gt_img)
            ocr_loss = self.mse_criterion(ocr_preds, ocr_gt)
        else:
            ocr_loss = torch.tensor(0.0).to(mse_loss.device)
        loss = (ce_loss + mse_loss) * self.alpha + ocr_loss * (1 - self.alpha)
        return {'loss': loss, 'mse_loss': mse_loss, 'ce_loss': ce_loss, 'ocr_loss': ocr_loss}, pred_latent

    def split_characters(self, pred, gt, indices):
        pred = self.vae.decode(pred).sample
        gt = self.vae.decode(gt).sample
        img = torch.cat([gt, pred], dim=-2)

        curr_char = indices[0]
        for idx, char in enumerate(indices):
            if char != curr_char:
                img[:, :, :, idx * 8 - 1] = -1
                curr_char = char

        img = self.write_text_below_image(img, self.tokenizer.decode(indices))

        return img
    

    @torch.no_grad()
    def write_text_below_image(self, image, text):
        image = (torch.clamp(image, -1, 1) + 1) * 127.5
        image = rearrange(image.to(torch.uint8), '1 1 h w -> h w').cpu().numpy()
        image = Image.fromarray(image, mode='L')

        text = text.replace('<pad>', '#').replace('</s>', '$')

        # Load the font
        font = ImageFont.load_default()
        ascent, descent = font.getmetrics()
        (width, baseline), (offset_x, offset_y) = font.font.getsize(text)

        # Calculate dimensions for the new image
        img_width, img_height = image.size
        new_height = img_height + offset_y + ascent +descent

        # Create a new image with white background
        new_image = Image.new('L', (img_width, new_height), color='white')

        # Paste the original image onto the new image
        new_image.paste(image, (0, 0))

        # Draw the text onto the new image
        draw = ImageDraw.Draw(new_image)

        curr_char = None
        for idx, char in enumerate(text):
            if char != curr_char:
                curr_char = char
                draw.text((idx * 8, img_height), char, fill='black', font=font)

        return new_image
    
    @torch.inference_mode()
    def generate_batch(self, decoder_inputs_embeds_vae, style_text, gen_text, cfg_scale=1.0,
                       max_new_tokens=64, min_gen_tokens=0, prefix_lens=None):
        """배치 + KV 캐시 자기회귀 생성.

        원본 구현은 매 토큰마다 (a) T5 encoder 를 텍스트에 다시 돌리고 (b) 디코더를
        prefix 전체에 다시 돌렸다 — 스텝당 O(n), 전체 O(n²). 여기선 encoder 출력과
        decoder KV 를 캐시해 스텝당 새 토큰 1개만 계산한다(수치적으로 동일, fp 오차 ~1e-7).
        cfg_scale != 1.0 일 때도 uncond/cond 를 [2b] 한 배치로 합쳐 forward 1회로 처리.

        Args:
            decoder_inputs_embeds_vae: [b, w, 8] style prefix latent (get_model_inputs 출력)
            prefix_lens: 샘플별 prefix 의 '실제' latent 길이. get_model_inputs 는 배치
                최대폭으로 오른쪽 패딩하므로 이걸 안 주면 패딩까지 style 로 먹는다.
                (여기선 좌측 패딩으로 다시 정렬 — T5 는 상대위치라 안전)
            min_gen_tokens: 이 토큰 수 전까지는 EOG(종료) 예측을 억제 (조기종료 방지 테스트용)

        Returns:
            (imgs, specials) — imgs[i]=[c,h,w] 라인이미지([-1,1]),
            specials[i]=1-D (3=style img, 2=image, 0=SOG, 1=EOG). 샘플마다 길이가 다르다.
        """
        device = self.T5.device
        z = decoder_inputs_embeds_vae.to(device)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        bs, max_w = z.size(0), z.size(1)
        if prefix_lens is None:
            lens = [max_w] * bs
        else:
            lens = [max(1, min(int(el), max_w)) for el in prefix_lens]

        # 텍스트 인코딩 — 루프 밖에서 한 번만 (원본은 매 토큰 encoder 재계산)
        _ids, _am = self._encode_text(style_text, gen_text)
        text_input_ids = _ids.to(device)
        text_mask = _am.to(device)

        use_cfg = cfg_scale != 1.0
        cond_text_embeds = self.T5.shared(text_input_ids)
        if use_cfg and self.drop_text:
            uncond_text_embeds = torch.zeros_like(cond_text_embeds) + self.uncond_embedding.weight
        else:
            uncond_text_embeds = cond_text_embeds

        # --- decoder prefix: sos + query_emb(style latent) [+ sog], 샘플별 길이가 달라 좌측 패딩 ---
        sos, sog = self.sos.weight[0], self.sog.weight[0]
        emb = self.query_emb(z)
        prefix_parts, spec_head, z_head = [], [], []
        for el in range(bs):
            parts = [sos[None], emb[el, :lens[el]]]
            specs = [torch.full((lens[el],), 3.0, device=device)]
            zs = [z[el, :lens[el]]]
            if len(style_text[el]) == 0:
                # 원본과 동일: style 텍스트가 없으면 style→gen 경계 SOG 를 미리 넣어준다
                parts.append(sog[None])
                specs.append(torch.zeros(1, device=device))
                zs.append(self.t5_to_vae(sog[None]))
            prefix_parts.append(torch.cat(parts, dim=0))
            spec_head.append(torch.cat(specs))
            z_head.append(torch.cat(zs, dim=0))

        plens = [p.size(0) for p in prefix_parts]
        pmax, d_model = max(plens), prefix_parts[0].size(-1)
        prefix = emb.new_zeros(bs, pmax, d_model)
        dec_mask = torch.zeros(bs, pmax, dtype=torch.long, device=device)
        for el, p in enumerate(prefix_parts):
            prefix[el, pmax - plens[el]:] = p
            dec_mask[el, pmax - plens[el]:] = 1

        if use_cfg:
            if self.drop_img:
                # 원본은 매 스텝 디코더 입력 '전체'를 uncond 로 갈아끼웠는데, KV 캐시 경로는
                # prefix 를 한 번 넣고 이후엔 새 토큰만 넣으므로 그 치환을 재현할 수 없다.
                # 이 repo 는 drop_img 를 켜지 않는다(항상 False) — 조용히 갈리느니 막는다.
                raise NotImplementedError("generate_batch 는 drop_img=True + CFG 를 지원하지 않는다")
            enc_embeds = torch.cat([uncond_text_embeds, cond_text_embeds], dim=0)
            enc_mask = torch.cat([text_mask, text_mask], dim=0)
            dec_in = torch.cat([prefix, prefix], dim=0)
            cur_mask = torch.cat([dec_mask, dec_mask], dim=0)
        else:
            enc_embeds, enc_mask = cond_text_embeds, text_mask
            dec_in, cur_mask = prefix, dec_mask

        encoder_outputs = self.T5.get_encoder()(inputs_embeds=enc_embeds, attention_mask=enc_mask)

        past, step_lat, step_spec = None, [], []
        finished = torch.zeros(bs, dtype=torch.bool, device=device)
        for i in range(max_new_tokens):
            out = self.T5(encoder_outputs=encoder_outputs, attention_mask=enc_mask,
                          decoder_inputs_embeds=dec_in, decoder_attention_mask=cur_mask,
                          past_key_values=past, use_cache=True)
            past = out.past_key_values
            output = out.logits[:, -1:]
            if use_cfg:
                uncond, cond = output[:bs], output[bs:]
                output = uncond + (cond - uncond) * cfg_scale

            special_prediction = self.t5_to_special(output)
            if i < min_gen_tokens:
                # 조기종료 방지: 일정 길이 전까지 EOG(index 1) 예측 억제
                special_prediction = special_prediction.clone()
                special_prediction[..., 1] = -1e9
            choice = special_prediction.argmax(dim=-1)[:, 0]         # [bs] 0=SOG 1=EOG 2=img

            vae_latent = self.t5_to_vae(output)                      # [bs, 1, 8]
            step_lat.append(vae_latent)
            step_spec.append(choice)
            finished |= (choice == 1)
            if bool(finished.all()):
                break

            # SOG 를 뽑았으면 다음 입력은 sog 임베딩, 아니면 방금 뽑은 latent (원본과 동일)
            nxt = torch.where((choice == 0)[:, None, None],
                              sog.view(1, 1, -1).expand(bs, 1, d_model),
                              self.query_emb(vae_latent))
            dec_in = torch.cat([nxt, nxt], dim=0) if use_cfg else nxt
            ones = torch.ones(cur_mask.size(0), 1, dtype=cur_mask.dtype, device=device)
            cur_mask = torch.cat([cur_mask, ones], dim=1)

        lats = torch.cat(step_lat, dim=1)                            # [bs, t, 8]
        specs = torch.stack(step_spec, dim=1)                        # [bs, t]
        imgs, specials = [], []
        for el in range(bs):
            eog = (specs[el] == 1).nonzero().flatten()
            n = int(eog[0].item()) + 1 if len(eog) else specs.size(1)   # EOG 토큰까지 포함
            z_seq = torch.cat([z_head[el], lats[el, :n]], dim=0)[None].to(self.vae.device)
            img = torch.clamp(self.vae.decode(self.z_rearrange(z_seq)).sample, -1, 1)
            imgs.append(img[0])
            specials.append(torch.cat([spec_head[el], specs[el, :n].to(spec_head[el].dtype)]))
        return imgs, specials

    @torch.inference_mode()
    def generate(self, decoder_inputs_embeds_vae, style_text, gen_text, cfg_scale=1.0, max_new_tokens=64, min_gen_tokens=0):
        """단일 샘플 생성. 배치는 generate_batch() 를 쓸 것.

        min_gen_tokens: 이 토큰 수 전까지는 EOG(종료) 예측을 억제 (조기종료 방지 테스트용)
        """
        if decoder_inputs_embeds_vae.dim() == 3 and decoder_inputs_embeds_vae.size(0) != 1:
            raise ValueError(f"generate() 는 bs=1 전용 (받은 bs={decoder_inputs_embeds_vae.size(0)}) "
                             "— 배치는 generate_batch() 를 쓸 것")
        imgs, specials = self.generate_batch(decoder_inputs_embeds_vae, style_text, gen_text,
                                             cfg_scale=cfg_scale, max_new_tokens=max_new_tokens,
                                             min_gen_tokens=min_gen_tokens)
        return imgs[0][None], specials[0]


    @torch.no_grad()
    def continue_gen_test(self, gt, batch, max_new_tokens=64, cfg_scale=1.0):
        gt = gt[:1]
        def _continue_gen(style_len):
            
            generation = self.generate(batch['decoder_inputs_embeds'][:1, :style_len], batch['style_text'][:1], batch['gen_text'][:1], cfg_scale=cfg_scale, max_new_tokens=max_new_tokens)
            test_img = generation[0]
            special_sequence = generation[1].repeat_interleave(8)

            
            special_img = torch.zeros_like(test_img).repeat(1,3,1,1)
            special_sequence = special_sequence[:special_img.size(-1)]
            special_img[:,0,:,special_sequence == 2] = 1 # red: image
            special_img[:,1,:,special_sequence == 0] = 1 # green: sog
            special_img[:,2,:,special_sequence == 1] = 1 # blue: eog
                        
            try:
                test_img[:, :, :, style_len * 8] = -1  # add a black line between style and pred
            except:
                print("couldn't add black line")
                # add special_img to the bottom of test_img
            test_img = torch.cat([test_img.repeat(1,3,1,1) , special_img], dim=-2)
            return test_img
        
        gt = torch.clamp(self.vae.decode(gt).sample, -1, 1)
        if type(batch['style_img_width']) == torch.Tensor:
            style_img_width = batch['style_img_width'][0]
        else:
            style_img_width = batch['style_img_width']

        return torch.cat(list(pad_images([
            # make_grid(_continue_gen(style_img_width//8-10), nrow=1, normalize=True),
            make_grid(_continue_gen(style_img_width//8), nrow=1, normalize=True),
        ])), dim=-2)


    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.T5.state_dict(), path / 'T5.pth')
        torch.save(self.vae.state_dict(), path / 'VAE.pth')
        if self.ocr is not None:   # ocr_checkpoint 없이 만들면 OCR 모듈 자체가 없다(기본값)
            torch.save(self.ocr.state_dict(), path / 'OCR.pth')
        torch.save(self.query_emb.state_dict(), path / 'query_emb.pth')
        torch.save(self.sos.state_dict(), path / 'sos.pth')

    def load_pretrained(self, path):
        path = Path(path)
        self.T5.load_state_dict(torch.load(path / 'T5.pth'))
        self.vae.load_state_dict(torch.load(path / 'VAE.pth'))
        if self.ocr is not None:
            self.ocr.load_state_dict(torch.load(path / 'OCR.pth'))
        self.query_emb.load_state_dict(torch.load(path / 'query_emb.pth'))
        self.sos.load_state_dict(torch.load(path / 'sos.pth'))

class DDPCompatibleEmuru(Emuru):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch_data, mode='train'):
        """
        Unified forward method that handles different modes for DDP compatibility
        """
        if mode == 'train':
            # Training mode - expects the full batch with model inputs already computed
            return super().forward(
                batch_data['decoder_inputs_embeds'], 
                batch_data['specials'], 
                batch_data['style_text'], 
                batch_data['gen_text']
            )
        elif mode == 'get_model_inputs':
            # Mode to get model inputs
            return super().get_model_inputs(
                batch_data['style_img'],
                batch_data['gen_img'], 
                batch_data['style_img_width'],
                batch_data['gen_img_width'],
                batch_data['max_img_len']
            )
        elif mode == 'generate':
            # Generation mode
            return super().generate(
                batch_data['decoder_inputs_embeds_vae'],
                batch_data['style_text'],
                batch_data['gen_text'],
                batch_data.get('cfg_scale', 1.0),
                batch_data.get('max_new_tokens', 64)
            )
        elif mode == 'continue_gen_test':
            # Continue generation test mode
            return super().continue_gen_test(
                batch_data['gt'],
                batch_data['batch'],
                max_new_tokens=batch_data.get('max_new_tokens', 64),
                cfg_scale=batch_data.get('cfg_scale', 1.0),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def module_get_model_inputs(self, style_img, gen_img, style_len, gen_len, max_img_len):
        """Direct access method for get_model_inputs when not using DDP forward"""
        return super().get_model_inputs(style_img, gen_img, style_len, gen_len, max_img_len)
    
    def module_continue_gen_test(self, gt, batch, max_new_tokens=64, cfg_scale=1.0):
        """Direct access method for continue_gen_test when not using DDP forward"""
        return super().continue_gen_test(gt, batch, max_new_tokens, cfg_scale)
    
    def module_vae_decode(self, latents):
        """Direct access method for VAE decode"""
        return self.vae.decode(latents)

    def get_trainable_parameters(self):
        """
        Get only the parameters that have requires_grad=True
        Useful for creating optimizers with only trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_parameter_count(self):
        """
        Get counts of total and trainable parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
    
    def print_parameter_info(self):
        """
        Print detailed information about model parameters
        """
        info = self.get_parameter_count()
        print(f"Model Parameter Info:")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        print(f"  Frozen parameters: {info['frozen_parameters']:,}")
        print(f"  Trainable ratio: {info['trainable_parameters']/info['total_parameters']:.2%}")
        
        # Print per-module info
        print(f"\nPer-module breakdown:")
        for name, module in self.named_children():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module_total > 0:
                print(f"  {name}: {module_trainable:,}/{module_total:,} trainable ({module_trainable/module_total:.1%})")
