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
from torchvision.transforms import Normalize
import numpy as np
import torch.nn as nn
from typing import Tuple

# Safer defaults for clearer NCCL/CUDA error reporting during debugging
_os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
_os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _safe_int_from_maybe_tensor(value, fallback_min: int = 64) -> int:
    """Convert a python int or 0-dim tensor (cpu/cuda) to int safely.

    - Synchronizes CUDA before .item() to surface the true failing kernel site
    - Moves to CPU before scalarization
    - Falls back to a reasonable minimum on unexpected errors
    """
    try:
        if isinstance(value, torch.Tensor):
            scalar_tensor = value
            # Take the first element if tensor is not scalar
            if scalar_tensor.dim() > 0:
                scalar_tensor = scalar_tensor.reshape(-1)[0]
            # Synchronize to attribute errors to the right op during debug
            if scalar_tensor.is_cuda:
                try:
                    torch.cuda.synchronize(scalar_tensor.device)
                except Exception:
                    pass
            return int(scalar_tensor.detach().to("cpu").item())
        return int(value)
    except Exception:
        # As a last resort, return a conservative minimum width
        return int(fallback_min)

def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    padded = rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')
    return padded.contiguous()




# sog, eog, img
SPECIAL_TOKEN_COUNT = 3

class Emuru(torch.nn.Module):
    def __init__(self, t5_checkpoint='google-t5/t5-base',
                 vae_checkpoint='blowing-up-groundhogs/emuru_vae',
                 ocr_checkpoint='files/checkpoints/Origami_bw_img/origami.pth', slices_per_query=1, channels=1, text_dropout_probability=0.0, img_dropout_probability=0.0):
        super(Emuru, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')  # per-character tokenizer
        self.tokenizer.add_tokens(["<sog>"])
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)
        self.t5_name_or_path = t5_checkpoint

        self.padding_token = torch.tensor([[-0.4951,  0.8021,  0.3429,  0.5622,  0.5271,  0.5756,  0.7194,  0.6150]])
        self.padding_token_threshold = 0.484982096850872

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
        self.normalize = Normalize(0.5, 0.5)
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

        self.ocr = OrigamiNet.from_checkpoint(ocr_checkpoint, o_classes=165, n_channels=1)
        self.set_training(self.ocr, False)
        
        self.query_rearrange = Rearrange('b c h (w q) -> b w (q c h)', q=slices_per_query)
        self.special_rearrange = torch.nn.Identity()
        # self.special_rearrange = Rearrange('b w (h c) -> b w (h c)')
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=channels, q=slices_per_query)
        self.z_rearrange_eval = Rearrange('w b (q c h) -> b c h (w q)', c=channels, q=slices_per_query)

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
        img = self.normalize(img)
        # Ensure contiguous memory layout before encode to avoid kernel issues
        img = img.contiguous()
        return self.vae.encode(img.float()).latent_dist.sample()

    @torch.no_grad()
    def get_model_inputs(self, style_img, gen_img, style_len, gen_len, max_img_len):
        bs = len(style_img)
        decoder_inputs_embeds_list = []
        specials_list = []
        
        # Move images to device and pad them
        style_img = pad_images([el.to(self.T5.device) for el in style_img])
        
        if gen_img is not None:
            gen_img = pad_images([el.to(self.T5.device) for el in gen_img])
            gen_img_embeds = self._img_encode(gen_img)
        else:
            gen_img_embeds = None

        style_img_embeds = self._img_encode(style_img)
        
        for el in range(bs):
            if isinstance(style_len, int):
                sl = style_len
            else:
                # Safely get scalar style length
                sl_tensor = style_len[el] if hasattr(style_len, '__getitem__') else style_len
                sl = _safe_int_from_maybe_tensor(sl_tensor)

            # Ensure widths are within bounds
            sl = max(64, min(sl, style_img_embeds.shape[-1]))
            
            # Start with style image embeds
            sample_embeds_parts = [style_img_embeds[el,:,:,:sl//8]]
            specials_parts = [torch.ones(sl//8) * 2] # Img token

            if gen_img_embeds is not None and gen_len is not None:
                if isinstance(gen_len, int):
                    gl = gen_len
                else:
                    gl_tensor = gen_len[el] if hasattr(gen_len, '__getitem__') else gen_len
                    gl = _safe_int_from_maybe_tensor(gl_tensor)

                gl = max(64, min(gl, gen_img_embeds.shape[-1]))
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
        }

    def forward(self, decoder_inputs_embeds_vae, specials, style_text, gen_text, ce_multiplier=1.0):
        # style_img_embeds: [bs, w//8, 8, 1]
        # generate text embeddings
        
        with torch.no_grad():
            encoded_text = self.tokenizer([f"{style}<sog>{gen}" for style, gen in zip(style_text, gen_text)], padding=True, return_tensors="pt")
        
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

        inputs_embeds = self.T5.shared(encoded_text['input_ids'].to(self.T5.device))
        drop_ids = torch.rand(inputs_embeds.shape[0], device=inputs_embeds.device) < self.dropout_probability
        if self.drop_text:
            inputs_embeds = torch.where(drop_ids[:, None, None], self.uncond_embedding.weight, inputs_embeds)
        if self.drop_img:
            decoder_inputs_embeds = torch.where(drop_ids[:, None, None], self.uncond_embedding.weight, decoder_inputs_embeds)
        
        output = self.T5(inputs_embeds=inputs_embeds, attention_mask=encoded_text['attention_mask'].to(self.T5.device), decoder_inputs_embeds=decoder_inputs_embeds)
        
        vae_latent = self.t5_to_vae(output.logits[:, :-1])
        special_latent = self.t5_to_special(output.logits[:, :-1]) # [bs, w//8, 3]
        pred_latent = self.z_rearrange(vae_latent)
        special_pred = self.special_rearrange(special_latent)

        
        ce_loss = ce_multiplier * self.ce_criterion(special_pred.flatten(0,1), specials.flatten(0,1))

        mse_mask = (specials == 2).unsqueeze(2) # [bs, w//8] TODO:consider putting the mask back in
        gt = decoder_inputs_embeds_vae * mse_mask
        vae_latent = vae_latent * mse_mask
        mse_loss = self.mse_criterion(vae_latent, gt)#/mse_mask.sum()
        ocr_loss = 0

        if self.alpha < 1.0:
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
    def generate(self, decoder_inputs_embeds_vae, style_text, gen_text, cfg_scale=1.0, max_new_tokens=64, min_gen_tokens=0):
        """
        call this with bs=1 please
        min_gen_tokens: 이 토큰 수 전까지는 EOG(종료) 예측을 억제 (조기종료 방지 테스트용)
        """
        encoded_text = self.tokenizer([f"{style}<sog>{gen}" for style, gen in zip(style_text,gen_text)], padding=True, return_tensors="pt")
        text_input_ids = encoded_text['input_ids'].to(self.T5.device)
        text_mask = encoded_text['attention_mask'].to(self.T5.device)

        sog = repeat(self.sog.weight, '1 d -> b 1 d', b=1)
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=1)
        z_sequence = [decoder_inputs_embeds_vae]
        special_sequence = torch.ones(decoder_inputs_embeds_vae.size(1))*3
        if len(z_sequence) == 0:
            decoder_inputs_embeds = sos
        else:
            decoder_inputs_embeds = self.query_emb(torch.cat(z_sequence, dim=1))
            if len(style_text[0]) != 0:
                decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
            else:
                decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds, sog], dim=1)
                vae_latent = self.t5_to_vae(sog)
                special_sequence = torch.cat([special_sequence, torch.zeros(1)])
                z_sequence.append(vae_latent)

        for i in range(max_new_tokens):
            if cfg_scale != 1.0:
                conditional_text_embeds = self.T5.shared(text_input_ids)
                if self.drop_text:
                    unconditional_text_embeds = torch.zeros_like(conditional_text_embeds).to(self.T5.device) + self.uncond_embedding.weight
                else:
                    unconditional_text_embeds = conditional_text_embeds

                if self.drop_img:
                    unconditional_decoder_inputs_embeds = torch.zeros_like(decoder_inputs_embeds).to(self.T5.device) + self.uncond_embedding.weight
                else:
                    unconditional_decoder_inputs_embeds = decoder_inputs_embeds

                output_unconditional = self.T5(inputs_embeds=unconditional_text_embeds, attention_mask=text_mask, decoder_inputs_embeds=unconditional_decoder_inputs_embeds).logits[:, -1:]
                output_conditional = self.T5(input_ids=text_input_ids, attention_mask=text_mask, decoder_inputs_embeds=decoder_inputs_embeds).logits[:, -1:]
                output = output_unconditional + (output_conditional - output_unconditional) * cfg_scale
            else:
                output = self.T5(input_ids=text_input_ids, attention_mask=text_mask, decoder_inputs_embeds=decoder_inputs_embeds).logits[:, -1:]

            special_prediction = self.t5_to_special(output)

            if i < min_gen_tokens:
                # 조기종료 방지: 일정 길이 전까지 EOG(index 1) 예측 억제
                special_prediction = special_prediction.clone()
                special_prediction[..., 1] = -1e9

            if torch.argmax(special_prediction, dim=-1) == 0:
                decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, sog], dim=1)
                vae_latent = self.t5_to_vae(output)
                special_sequence = torch.cat([special_sequence, torch.zeros(1)])
            elif torch.argmax(special_prediction, dim=-1) == 1:
                special_sequence = torch.cat([special_sequence, torch.ones(1)])
                vae_latent = self.t5_to_vae(output)
                z_sequence.append(vae_latent)
                break
            else:
                vae_latent = self.t5_to_vae(output)
                decoder_inputs_embeds = torch.cat([decoder_inputs_embeds, self.query_emb(vae_latent)], dim=1)
                special_sequence = torch.cat([special_sequence, torch.ones(1)*2])
            z_sequence.append(vae_latent)
            
            
        z_sequence = [el.to(self.vae.device) for el in z_sequence]
        
        z_sequence = torch.cat(z_sequence, dim=1)
        img = torch.clamp(self.vae.decode(self.z_rearrange(z_sequence)).sample, -1, 1)
        return img, special_sequence.to(self.T5.device)
        
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
        torch.save(self.ocr.state_dict(), path / 'OCR.pth')
        torch.save(self.query_emb.state_dict(), path / 'query_emb.pth')
        torch.save(self.sos.state_dict(), path / 'sos.pth')

    def load_pretrained(self, path):
        path = Path(path)
        self.T5.load_state_dict(torch.load(path / 'T5.pth'))
        self.vae.load_state_dict(torch.load(path / 'VAE.pth'))
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
                batch_data.get('cfg_scale', 1.0),
                batch_data.get('max_new_tokens', 64)
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
