import copy
import math
import random
from collections import OrderedDict
from dataclasses import asdict
from functools import partial
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch import nn
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.utils.checkpoint import checkpoint

from .rope import Rope3D
from .sincos_pe import get_3d_sincos_pos_embed
# from core.vision_encoder.config import PEConfig, PETextConfig, PE_VISION_CONFIG, PE_TEXT_CONFIG, fetch_pe_checkpoint

import os
logger = getLogger()

ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "TORCH_SDPA")
print(f"ATTENTION_BACKEND={ATTENTION_BACKEND}")
if ATTENTION_BACKEND == "FLASH_ATTN_V3":
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    except (ModuleNotFoundError, ImportError) as e:
        flash_attn_func_v3 = None
        print(f"FA3 not available: {e}")
    else:
        print("Using FA3")


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.dim = dim
        self.init_values = init_values

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    def init_tensors(self):
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))


class AttentionPooling(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_probe: int = 1,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert (
            self.embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        self.probe = nn.Parameter(torch.randn(1, num_probe, self.embed_dim))
        self.attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, batch_first=True
        )

        self.layernorm = norm_layer(embed_dim)
        self.mlp_width = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(self.embed_dim, self.mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(self.mlp_width, self.embed_dim)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        batch, _, _ = x.shape

        q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
        x = self.attn(q, x, x, need_weights=False)[0]
        x = x + self.mlp(self.layernorm(x))

        return x


class SelfAttention(nn.Module):
    r"""
    Implements sequence packed attention and RoPe
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: Optional[nn.Module] = None,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # To make this compatibile with nn.MultiHeadAttention
        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.scale = self.head_dim ** (-0.5)

    def init_tensors(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(self, x, attn_mask=None, mask=None):
        batch, seq, embed_dim = x.shape
        proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)

        # reshape to 3, E and not E, 3 is deliberate for better memory coalescing and keeping same order as chunk()
        proj = (
            proj.unflatten(-1, (3, embed_dim))
            .unsqueeze(0)
            .transpose(0, -2)
            .squeeze(-2)
            .contiguous()
        )
        q, k, v = proj[0], proj[1], proj[2]

        # Use "q_" so that we don't accidentally quit in pdb :)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if self.rope:
            q, k = self.rope(q, k, mask=mask)

        # attn = F.scaled_dot_product_attention(
        #     q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        # )
        if ATTENTION_BACKEND == "TORCH_SDPA":
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
            )
            attn = rearrange(attn, "b h s d -> b s (h d)")
        elif ATTENTION_BACKEND == "FLASH_ATTN_V3":
            # print(f"qkv: {q.shape}, {k.shape}, {v.shape}")
            attn = flash_attn_func_v3(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=False,
                softmax_scale=self.scale,
            )   # 新版这里没有多余输出
            if isinstance(attn, tuple):
                attn = attn[0]
            attn = rearrange(attn, "b s h d -> b s (h d)")
        else:
            raise Exception(f"ATTENTION_BACKEND:{ATTENTION_BACKEND} not support.")
        
        

        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        if rope:
            self.attn = SelfAttention(d_model, n_head, rope=rope)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def _call_attn(
        self,
        q_x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        mask = None
    ):

        if attn_mask is not None:
            # Leave boolean masks as is
            if not attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q_x.dtype)

        if isinstance(self.attn, SelfAttention):
            return self.attn(q_x, attn_mask=attn_mask, mask=mask)
        else:
            # raise ValueError("Not implement !!!")
            return self.attn(q_x, q_x, q_x, attn_mask=attn_mask, need_weights=False)[0]

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        mask = None
    ):
        x = x + self.drop_path1(
            self.ls_1(self._call_attn(self.ln_1(x), attn_mask=attn_mask, mask=mask))
        )
        x = x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    rope=rope,
                )
                for _ in range(layers)
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def truncate(self, layer_idx: int):
        """ Delete layers so the last layer is the given layer index. """
        self.layers = ((self.layers + layer_idx) % self.layers) + 1
        self.resblocks = nn.ModuleList(self.resblocks[:self.layers])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        mask = None,
    ):
        stop_idx = (self.layers + layer_idx) % self.layers

        for i, r in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, attn_mask, mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask, mask=mask)
            
            if i == stop_idx:
                break

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5),
        use_ln_pre: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float = None,
        drop_path: float = 0.0,
        image_size: int = 448,  # Pretrain image size only; you can pass in any image size
        use_abs_posemb: bool = True,
        use_rope: bool = True,
        use_cls_token: bool = False,
        output_dim: Optional[int] = 1280,
        attn_pooler_heads: int = 8,
        pool_type: Literal["attn", "tok", "avg", "none"] = "attn",

        # decoder_width = 512,
        # decoder_layers = 8,
        # decoder_heads = 16,
        # norm_pix_loss = True,

        device='cuda',
        # add params
        depth_patch_size: int = None,
    ):
        super().__init__()
        assert pool_type in ("attn", "tok", "avg", "none")
        self.pool_type = pool_type
        self.patch_size = patch_size
        self.depth_patch_size = depth_patch_size

        self.output_dim = output_dim or width
        self.proj_dim = output_dim
        self.heads = heads
        self.width = width
        self.layers = layers

        self.use_abs_posemb = use_abs_posemb
        self.use_cls_token = use_cls_token
        self.use_rope3d = use_rope
        self.image_size = image_size
        self.grad_checkpoint = False
        # self.norm_pix_loss = norm_pix_loss

        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.rope = (
            Rope3D(
                dim=width // heads,
                use_cls_token=self.use_cls_token,
                device=device
            )
            if self.use_rope3d
            else None
        )

        self.ln_pre = norm_layer(width) if use_ln_pre else nn.Identity()
        self.ln_post = norm_layer(self.width) if use_ln_post else nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
        )

        if pool_type == "attn":
            self.attn_pool = AttentionPooling(
                embed_dim=width,
                num_heads=attn_pooler_heads,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None

        ## mae
        # self.mask_token = nn.Parameter(torch.zeros(1, decoder_width))

        # num_patches = (self.image_size // self.patch_size) ** 3
        # self.num_patches = num_patches
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + int(self.use_cls_token), width), requires_grad=False)
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + int(self.use_cls_token), decoder_width), requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_rope = (
        #     Rope3D(
        #         dim=decoder_width // decoder_heads,
        #         use_cls_token=self.use_cls_token,
        #         device=device
        #     )
        #     if self.use_rope3d
        #     else None
        # )

        # self.decoder_embed = nn.Linear(width, decoder_width, bias=True)

        # self.decoder = Transformer(
        #     decoder_width,
        #     decoder_layers,
        #     decoder_heads,
        #     mlp_ratio,
        #     ls_init_value=ls_init_value,
        #     act_layer=act_layer,
        #     norm_layer=norm_layer,
        #     drop_path=drop_path,
        #     rope=self.decoder_rope,
        # )

        # self.decoder_pred = nn.Linear(decoder_width, patch_size**3 * 1, bias=True)

        # self.decoder_post = norm_layer(decoder_width)

        self.init_tensors()
        
        print("pool type:", self.pool_type)
        print("rope:", self.use_rope3d)


    def init_tensors(self):
        def init_submodule_tensors(module):
            for name, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    logger.debug(f"Initializing tensors for submodule: {name}")
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)
        if self.rope is not None:
            self.rope.init_tensors()
            # self.decoder_rope.init_tensors()

        # class embeddings and positional embeddings
        init_scale = self.width ** -0.5

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))

        if self.use_abs_posemb:
            # image_size 应该要给128
            assert self.image_size == 128
            self.posemb_grid_size = self.image_size // self.patch_size
            self.positional_embedding = nn.Parameter(
                init_scale
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**3, self.width
                    # pos_embed.shape[-1]
                )
            )

            # self.decoder_positional_embedding = nn.Parameter(
            #     init_scale
            #     * torch.randn(
            #         int(self.use_cls_token) + self.posemb_grid_size**3, self.decoder_pos_embed.shape[-1]
            #     )
            # ) 
            # pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(round(self.num_patches**(1/3))), cls_token=self.use_cls_token)
            # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(round(self.num_patches**(1/3))), cls_token=self.use_cls_token)
            # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # torch.nn.init.normal_(self.mask_token, std=.02)

        if self.proj_dim is not None:
            self.proj = nn.Parameter(
                init_scale * torch.randn(self.width, self.proj_dim)
            )

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 1, H, W, D)
    #     x: (N, L, patch_size**3 *1)
    #     """
    #     p = self.patch_size
    #     assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

    #     h = w = d = imgs.shape[2] // p
    #     x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
    #     x = torch.einsum('nchpwqdr->nhwdpqrc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * 1))
    #     return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**3 *1)
    #     imgs: (N, 1, H, W, D)
    #     """
    #     p = self.patch_size
    #     h = w = d = int(round(x.shape[1]**(1/3)))
    #     assert h * w *d == x.shape[1]
        
    #     x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, 1))
    #     x = torch.einsum('nhwdpqrc->nchpwqdr', x)
    #     imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p, h * p))
    #     return imgs
    
    # def random_window_masking(self, x, mask_ratio, D,H,W, r=2):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     B, L, C = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))
        
    #     # 验证尺寸兼容性
    #     s_orig = H  # 原始空间尺寸
    #     assert H == W == D, "输入必须是立方体数据"
    #     assert s_orig % r == 0, f"尺寸{s_orig}必须能被分块大小{r}整除"
    #     d = s_orig // r  # 下采样后尺寸
        
    #     # 生成随机噪声并排序
    #     noise = torch.rand(B, d**3, device=x.device)
    #     sparse_shuffle = torch.argsort(noise, dim=1)
    #     sparse_restore = torch.argsort(sparse_shuffle, dim=1)
    #     keep_num_down = max(1, int(d**3 * (1 - mask_ratio)))  # 至少保留1个区块
    #     sparse_keep = sparse_shuffle[:, :keep_num_down]
        
    #     # 计算基础索引
    #     I = sparse_keep // (d * d)
    #     J = (sparse_keep // d) % d
    #     K = sparse_keep % d
    #     base = r * (I * (s_orig**2) + J * s_orig + K)  # [B, keep_num_down]
        
    #     # 生成立方体偏移量
    #     ii, jj, kk = torch.meshgrid(
    #         torch.arange(r, device=x.device),
    #         torch.arange(r, device=x.device),
    #         torch.arange(r, device=x.device),
    #         indexing='ij'
    #     )
    #     offsets = ii * (s_orig**2) + jj * s_orig + kk  # [r, r, r]
    #     offsets = offsets.reshape(-1)  # [r**3]
        
    #     # 计算保留的token索引
    #     indices_keep = base[..., None] + offsets  # [B, keep_num_down, r**3]
    #     indices_keep = indices_keep.reshape(B, -1)  # [B, keep_num_down * r**3]
        
    #     # 生成掩码索引
    #     indices_all = torch.arange(L, device=x.device).expand(B, -1)
    #     mask = torch.ones(B, L, dtype=torch.bool, device=x.device)
    #     # print(mask.shape, indices_keep.shape, indices_keep.max(), indices_keep.min())
    #     mask.scatter_(1, indices_keep, False)
    #     indices_mask = indices_all[mask].reshape(B, -1)
        
    #     # 创建恢复索引
    #     indices_shuffle = torch.cat([indices_keep, indices_mask], dim=1)
    #     indices_restore = torch.argsort(indices_shuffle, dim=1)

    #     x_masked = torch.gather(x, dim=1, index=indices_keep.unsqueeze(-1).repeat(1, 1, C))

    #     mask = torch.ones([B, L], device=x.device)
    #     mask[:, :indices_keep.shape[-1]] = 0
    #     mask = torch.gather(mask, dim=1, index=indices_restore)

    #     return x_masked, mask, indices_restore

    # def random_masking(self, x, mask_ratio):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))
        
    #     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore
    
    # def truncate(self, layer_idx: int):
    #     """ Delete layers so the last layer is the given layer index. """
    #     self.transformer.truncate(layer_idx)
    #     self.layers = self.transformer.layers


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpoint = enable
        self.transformer.set_grad_checkpointing(enable=enable)

    def _sample_abs_posemb(self, grid_z: int, grid_h: int, grid_w: int):
        """Interpolates the absolute position embedding if necessary."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w\
            and self.posemb_grid_size == grid_z:
            return self.positional_embedding[None, ...]

        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed, size=(grid_z, grid_h, grid_w), mode="trilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).reshape(-1, self.width).contiguous()

        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

        return pos_embed[None, ...]
    
    # def _decoder_sample_abs_posemb(self, grid_z: int, grid_h: int, grid_w: int):
    #     """Interpolates the absolute position embedding if necessary."""
    #     if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w\
    #         and self.posemb_grid_size == grid_z:
    #         return self.decoder_positional_embedding[None, ...]

    #     pos_embed = self.decoder_positional_embedding
    #     if self.use_cls_token:
    #         cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]

    #     pos_embed = (
    #         pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, self.posemb_grid_size, -1)
    #         .permute(0, 4, 1, 2, 3)
    #         .contiguous()
    #     )
    #     pos_embed = F.interpolate(
    #         pos_embed, size=(grid_z, grid_h, grid_w), mode="trilinear", align_corners=False
    #     )
    #     pos_embed = pos_embed.permute(0, 2, 3, 4, 1).reshape(-1, self.width).contiguous()

    #     if self.use_cls_token:
    #         pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)

    #     return pos_embed[None, ...]


    def forward_encoder(self, x, mask_ratio=0.0):
        # embed patches
        batch, _, z, h, w = x.shape
        grid_z, grid_h, grid_w = z // self.depth_patch_size, h // self.patch_size, w // self.patch_size

        # import ipdb; ipdb.set_trace()
        x = self.patch_embed(x) # [4, 1, 64, 128, 128] -> [4, 1024, 8, 16, 16]
        x = x.permute(0, 2, 3, 4, 1).reshape(batch, -1, self.width) # [4, 1024, 8, 16, 16] -> [4, 2048, 1024]

        # add pos embed w/o cls token
        x = x + self._sample_abs_posemb(grid_z, grid_h, grid_w) #self.pos_embed[:, int(self.use_cls_token):, :]
        # x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = (
        #     self.random_masking(x, mask_ratio)
        #     if self.patch_size >=16 else
        #     self.random_window_masking(x, mask_ratio=mask_ratio, D=grid_z,H=grid_h,W=grid_w, r=16 // self.patch_size)
        # )
        # append cls token
        # 没用
        assert not self.use_cls_token
        # if self.use_cls_token:
        #     cls_token = self.class_embedding + self.pos_embed[:, :1, :]
        #     cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        #     x = torch.cat((cls_tokens, x), dim=1)
        
        if self.use_rope3d:
            # if self.grad_checkpoint:
            #     self.rope.update_grid(x.device, 64, 64, 64) ### 1024 / 16
            # else:
            self.rope.update_grid(x.device, grid_z, grid_h, grid_w)
        
        # apply Transformer blocks
        x = self.ln_pre(x)
        x = self.transformer(x, layer_idx=-1, mask=None)
        x = self.ln_post(x)

        # print(f"after mae vit: {x.shape}")

        return x

    # def forward_decoder(self, x, ids_restore, grids=None):
    #     # embed tokens
    #     x = self.decoder_embed(x)

    #     # append mask tokens to sequence
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #     x_ = torch.cat([x[:, int(self.use_cls_token):, :], mask_tokens], dim=1)  # no cls token
    #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     x = torch.cat([x[:, :int(self.use_cls_token), :], x_], dim=1)  # append cls token

    #     # add pos embed
    #     x = x + self._decoder_sample_abs_posemb(*grids) #self.decoder_pos_embed[:, int(self.use_cls_token):, :]

    #     if self.use_rope3d:
    #         if self.grad_checkpoint:
    #             self.decoder_rope.update_grid(x.device, 64, 64, 64) ### 1024 / 16
    #         else:
    #             self.decoder_rope.update_grid(x.device, *grids)

    #     # apply Transformer blocks
    #     x = self.decoder(x, layer_idx=-1)
    #     x = self.decoder_post(x)

    #     # predictor projection
    #     x = self.decoder_pred(x)

    #     # remove cls token
    #     x = x[:, int(self.use_cls_token):, :]

    #     return x
    
    # def forward_loss(self, imgs, pred, mask):
    #     """
    #     imgs: [N, 1, H, W, D]
    #     pred: [N, L, p*p*p*1]
    #     mask: [N, L], 0 is keep, 1 is remove, 
    #     """
    #     target = self.patchify(imgs)
    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1.e-6)**.5
        
    #     # print(pred.shape, target.shape)
    #     loss = (pred - target) ** 2
    #     loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    #     loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    #     return loss

    # def forward(self, imgs, mask_ratio=0.0):
    #     # latent, mask, ids_restore, grids = self.forward_encoder(imgs, mask_ratio)
    #     # pred = self.forward_decoder(latent, ids_restore, grids=grids)  # [N, L, p*p*p*1]
    #     # loss = self.forward_loss(imgs, pred, mask)
    #     # return loss, pred, mask
    #     return self.forward_encoder(x=imgs, mask_ratio=mask_ratio)

    def _pool(self, x: torch.Tensor):
        if self.pool_type == "tok":
            return x[:, 0]
        elif self.pool_type == "avg":
            return x.mean(dim=1)
        elif self.pool_type == "attn":
            return self.attn_pool(x).squeeze(1)
        elif self.pool_type == "none":
            return x
        else:
            raise NotImplementedError

    # def forward_list(
    #     self, xs, masks=None, **kwargs
    # ):
    #     outs = []
    #     for x,mask in zip(xs, masks):
    #         y = self.forward(x, mask, **kwargs)
    #         outs.append(y)
    #     return outs
    
    # def forward_features(
    #     self,
    #     x: torch.Tensor,
    #     norm: bool = False,
    #     layer_idx: int = -1,
    #     strip_cls_token: bool = False,
    #     masks = None
    # ): 
    #     batch, _, z, h, w = x.shape
    #     grid_z, grid_h, grid_w = z // self.patch_size, h // self.patch_size, w // self.patch_size

    #     x = self.patch_embed(x)
    #     x = x.permute(0, 2, 3, 4, 1).reshape(batch, -1, self.width)

    #     if masks is not None:
    #         # print(masks.shape, self.mask_token.shape, x.shape)
    #         x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

    #     if self.use_cls_token:
    #         x = torch.cat(
    #             [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
    #             dim=1,
    #         )

    #     if self.use_abs_posemb:
    #         x = x + self._sample_abs_posemb(grid_z, grid_h, grid_w)

    #     if self.use_rope3d:
    #         if self.grad_checkpoint:
    #             self.rope.update_grid(x.device, 64, 64, 64) ### 1024 / 16
    #         else:
    #             self.rope.update_grid(x.device, grid_z, grid_h, grid_w)

    #     x = self.ln_pre(x)
    #     x = self.transformer(x, layer_idx=layer_idx)

    #     if norm:
    #         x = self.ln_post(x)

    #     if strip_cls_token and self.use_cls_token:
    #         x = x[:, 1:, :]

    #     return x

    def forward(self, x: torch.Tensor, **kwargs):
        
        # import ipdb; ipdb.set_trace()
        # x = self.forward_features(x, norm=True, masks=mask, **kwargs)
        x = self.forward_encoder(x, mask_ratio=0.0, **kwargs)
        x = self._pool(x)

        if self.proj_dim is not None:
            x = x @ self.proj

        return x
    
def vit_base(
    image_size=224, 
    patch_size=16,
    depth_patch_size=16,
    width=768,
    layers=12,
    heads=12,
    mlp_ratio=4.0,
    pool_type="none",
    output_dim=None,
    use_cls_token=True,
    drop_path = 0.0,
    use_rope = False,

    # decoder_width = 512,
    # decoder_layer = 8,
    # decoder_heads = 16,

    **kwargs
):
    ls_init_value: float = None
    # drop_path: float = 0.0

    # image_size: int = 224
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    
    # use_rope3d: bool = True
    # pool_type: str = "attn"
    
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True
    
    model = VisionTransformer(
        image_size = image_size,
        patch_size=patch_size,
        depth_patch_size=depth_patch_size,
        width=width,
        layers=layers,
        heads=heads,
        mlp_ratio=mlp_ratio,
        pool_type=pool_type,
        output_dim=output_dim,
        use_cls_token=use_cls_token,
        ls_init_value=ls_init_value,
        drop_path=drop_path,

        use_abs_posemb=use_abs_posemb,
        use_rope=use_rope,

        attn_pooler_heads=attn_pooler_heads,

        use_ln_pre=use_ln_pre,
        use_ln_post=use_ln_post,

        # decoder_width = 512,
        # decoder_layers = 8,
        # decoder_heads = 16,
        **kwargs,
    )
    return model
    

def vit_large(
    image_size=336, 
    patch_size=16,
    depth_patch_size=16,
    width=1024,
    layers=24,
    heads=16,
    mlp_ratio=4.0,
    pool_type="none",
    output_dim=None,
    use_cls_token=True,
    drop_path = 0.0,
    use_rope = False,

    # decoder_width = 512,
    # decoder_layer = 8,
    # decoder_heads = 16,
    return_config = False,
    **kwargs
):
    ls_init_value: float = None
    # drop_path: float = 0.0

    # image_size: int = 224
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    
    # use_rope3d: bool = True
    # pool_type: str = "attn"
    
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True
    
    model = VisionTransformer(
        image_size = image_size,
        patch_size=patch_size,
        depth_patch_size=depth_patch_size,
        width=width,
        layers=layers,
        heads=heads,
        mlp_ratio=mlp_ratio,
        pool_type=pool_type,
        output_dim=output_dim,
        use_cls_token=use_cls_token,
        ls_init_value=ls_init_value,
        drop_path=drop_path,

        use_abs_posemb=use_abs_posemb,
        use_rope=use_rope,

        attn_pooler_heads=attn_pooler_heads,

        use_ln_pre=use_ln_pre,
        use_ln_post=use_ln_post,

        # decoder_width = 512,
        # decoder_layers = 8,
        # decoder_heads = 16,
        **kwargs,
    )

    ### Add by hangjie
    if not return_config:
        return model, None
    else:
        config_dict = {
            "image_size": image_size,
            "patch_size": patch_size,
            "depth_patch_size": depth_patch_size,
            "width": width,
            "layers": layers,
            "heads": heads,
            "mlp_ratio": mlp_ratio,
            "pool_type": pool_type,
            "output_dim": output_dim,
            "use_cls_token": use_cls_token,
            "ls_init_value": ls_init_value,
            "drop_path": drop_path,
            "use_abs_posemb": use_abs_posemb,
            "use_rope": use_rope,
            "attn_pooler_heads": attn_pooler_heads,
            "use_ln_pre": use_ln_pre,
            "use_ln_post": use_ln_post,
        }
        return model, config_dict

def vit_giant(
    image_size=448, 
    patch_size=16,
    width=1536,
    layers=50,
    heads=16,
    mlp_ratio=8960 / 1536,
    pool_type="none",
    output_dim=None,
    use_cls_token=False,
    drop_path = 0.0,
    use_rope = False,

    decoder_width = 512,
    decoder_layer = 8,
    decoder_heads = 16,

    **kwargs
):
    ls_init_value: float = None
    # drop_path: float = 0.0

    # image_size: int = 224
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    
    # use_rope3d: bool = True
    # pool_type: str = "attn"
    
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True
    
    model = VisionTransformer(
        image_size = image_size,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=heads,
        mlp_ratio=mlp_ratio,
        pool_type=pool_type,
        output_dim=output_dim,
        use_cls_token=use_cls_token,
        ls_init_value=ls_init_value,
        drop_path=drop_path,

        use_abs_posemb=use_abs_posemb,
        use_rope=use_rope,

        attn_pooler_heads=attn_pooler_heads,

        use_ln_pre=use_ln_pre,
        use_ln_post=use_ln_post,

        decoder_width = 512,
        decoder_layers = 8,
        decoder_heads = 16,
        **kwargs,
    )
    return model
    
