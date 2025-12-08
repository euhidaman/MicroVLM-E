"""
DiET (Distilled Image Encoder with Transformers) Vision Encoder for MicroVLM-E.

Implements DiET-Tiny and DiET-Small variants based on the DeiT architecture.
DiET is an efficient vision transformer that can be used as a frozen visual encoder.
"""

import math
import logging
from functools import partial
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from microvlm_e.models.base_model import LayerNorm


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, E, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        return x


class Attention(nn.Module):
    """Multi-head Self Attention with optional QK-Norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # QK-Norm for training stability
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LayerNorm(head_dim)
            self.k_norm = LayerNorm(head_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, N, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK-Norm if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DiETEncoder(nn.Module):
    """
    DiET (Distilled Image Encoder Transformer) Vision Encoder.

    Based on DeiT architecture with distillation token support.
    Supports DiET-Tiny and DiET-Small configurations.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        use_distillation: bool = True,
        qk_norm: bool = False,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.use_distillation = use_distillation
        self.use_grad_checkpoint = use_grad_checkpoint

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS and distillation tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_distillation:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.num_tokens = 2
        else:
            self.dist_token = None
            self.num_tokens = 1

        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        # Initialize other weights
        self.apply(self._init_weights_module)

    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        """Extract features from image."""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS and distillation tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.use_distillation:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        """Forward pass returning all features (for Q-Former)."""
        return self.forward_features(x)

    def get_output_dim(self):
        """Get output dimension."""
        return self.embed_dim


def create_diet_encoder(
    model_name: str = "diet_tiny",
    img_size: int = 224,
    drop_path_rate: float = 0.0,
    use_grad_checkpoint: bool = False,
    pretrained: bool = True,
    freeze: bool = True,
    qk_norm: bool = False,
) -> Tuple[DiETEncoder, nn.LayerNorm]:
    """
    Create DiET encoder with specified configuration.

    Args:
        model_name: Model variant ('diet_tiny' or 'diet_small').
        img_size: Input image size.
        drop_path_rate: Drop path rate.
        use_grad_checkpoint: Whether to use gradient checkpointing.
        pretrained: Whether to load pretrained weights.
        freeze: Whether to freeze the encoder.
        qk_norm: Whether to use QK normalization.

    Returns:
        encoder: DiET encoder.
        ln_vision: Layer normalization for vision features.
    """
    logging.info(f"Loading DiET encoder: {model_name}")

    # Model configurations
    configs = {
        "diet_tiny": {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "mlp_ratio": 4.0,
        },
        "diet_small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
        },
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(configs.keys())}")

    config = configs[model_name]

    encoder = DiETEncoder(
        img_size=img_size,
        patch_size=16,
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_path_rate=drop_path_rate,
        use_grad_checkpoint=use_grad_checkpoint,
        qk_norm=qk_norm,
    )

    # Load pretrained weights if available
    if pretrained:
        try:
            import timm
            # Try to load from timm (DeiT weights are compatible)
            timm_name = "deit_tiny_patch16_224" if model_name == "diet_tiny" else "deit_small_patch16_224"
            timm_model = timm.create_model(timm_name, pretrained=True)

            # Map weights
            state_dict = timm_model.state_dict()
            # Remove classification head
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}

            msg = encoder.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded pretrained weights for {model_name}")
            if msg.missing_keys:
                logging.info(f"Missing keys: {msg.missing_keys}")
        except Exception as e:
            logging.warning(f"Could not load pretrained weights: {e}")
            logging.warning("Using random initialization")

    # Layer normalization for vision features
    ln_vision = nn.LayerNorm(encoder.embed_dim)

    # Freeze encoder if requested
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        encoder.train = lambda self, mode=True: self

        for param in ln_vision.parameters():
            param.requires_grad = False
        ln_vision.eval()
        ln_vision.train = lambda self, mode=True: self

        logging.info("Froze vision encoder")

    logging.info(f"DiET encoder loaded: {model_name}, embed_dim={encoder.embed_dim}")

    return encoder, ln_vision

