# This file is a modified version of the original file from the diffusers package.
#https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vq_model.py

from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalVAEMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_outputs import BaseOutput
from diffusers.models.modeling_utils import ModelMixin

from models.vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder, VectorQuantizer

@dataclass
class VQEncoderOutput(BaseOutput):
    """
    Output of VQModel encoding method.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The encoded output sample from the last layer of the model.
    """

    latents: torch.Tensor



class AutoencoderQ(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        dropout: float = 0.1,
        sample_size: int = 768,
        num_vq_embeddings: int = 256,
        lookup_from_codebook=False,
        force_upcast=False,
        vq_embed_dim: Optional[int] = None,
        vertical_total_compression: bool = False
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
            dropout=dropout,
            vertical_total_compression=vertical_total_compression,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            dropout=dropout,
            vertical_total_compression=vertical_total_compression,
        )

        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(num_vq_embeddings, vq_embed_dim, beta=0.25, remap=None, sane_index_shape=False, legacy=False)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25


    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> VQEncoderOutput:
        h = self.encoder(x)
        h = self.quant_conv(h)

        if not return_dict:
            return (h,)

        return VQEncoderOutput(latents=h)

    @apply_forward_hook
    def decode(
        self, quant: torch.Tensor, return_dict: bool = True, shape=None
    ) -> Union[DecoderOutput, torch.Tensor]:
        
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2)

        if not return_dict:
            return dec

        return DecoderOutput(sample=dec)

    def forward(
        self, sample: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, ...]]:
        r"""
        The [`VQModel`] forward method.

        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.autoencoders.vq_model.VQEncoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vq_model.VQEncoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoders.vq_model.VQEncoderOutput`] is returned, otherwise a
                plain `tuple` is returned.
        """

        h = self.encode(sample).latents
        dec = self.decode(h)

        if not return_dict:
            return dec.sample, dec.commit_loss
        return dec