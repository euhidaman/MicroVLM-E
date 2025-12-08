"""
Q-Former (Querying Transformer) for MicroVLM-E.

Based on BLIP-2's Q-Former architecture that bridges vision and language models.
Uses learned query tokens to extract visual features relevant for language generation.
"""

import math
import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertEmbeddings,
    BertPooler,
)


class QFormerEmbeddings(nn.Module):
    """
    Embeddings for Q-Former with query support.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        query_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if seq_length > 0:
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class QFormerSelfAttention(nn.Module):
    """Self-attention with optional cross-attention for Q-Former."""

    def __init__(self, config: BertConfig, is_cross_attention: bool = False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({config.hidden_size}) is not divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)

        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_cross_attention = is_cross_attention

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.is_cross_attention and encoder_hidden_states is not None:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply attention mask
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        return context_layer


class QFormerSelfOutput(nn.Module):
    """Output projection for Q-Former self-attention."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerAttention(nn.Module):
    """Attention layer for Q-Former with optional cross-attention."""

    def __init__(self, config: BertConfig, is_cross_attention: bool = False):
        super().__init__()
        self.self = QFormerSelfAttention(config, is_cross_attention)
        self.output = QFormerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class QFormerIntermediate(nn.Module):
    """Intermediate layer for Q-Former."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class QFormerOutput(nn.Module):
    """Output layer for Q-Former."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerLayer(nn.Module):
    """A single Q-Former layer with self-attention and optional cross-attention."""

    def __init__(self, config: BertConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Self-attention
        self.attention = QFormerAttention(config)

        # Cross-attention (only at certain layers)
        self.has_cross_attention = (
            config.add_cross_attention and
            layer_idx % config.cross_attention_freq == 0
        )
        if self.has_cross_attention:
            self.crossattention = QFormerAttention(config, is_cross_attention=True)

        # Feed-forward
        self.intermediate = QFormerIntermediate(config)
        self.output = QFormerOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        query_length: int = 0,
    ):
        # Self-attention
        self_attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
        )

        # Cross-attention (only on query tokens)
        if self.has_cross_attention and encoder_hidden_states is not None:
            query_attention_output = self.crossattention(
                self_attention_output[:, :query_length, :],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            # Concatenate query output with text output
            if query_length < self_attention_output.size(1):
                attention_output = torch.cat(
                    [query_attention_output, self_attention_output[:, query_length:, :]],
                    dim=1
                )
            else:
                attention_output = query_attention_output
        else:
            attention_output = self_attention_output

        # Feed-forward
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class QFormerEncoder(nn.Module):
    """Q-Former encoder consisting of multiple layers."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([
            QFormerLayer(config, i) for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        query_length: int = 0,
    ):
        for layer in self.layer:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                query_length=query_length,
            )
        return hidden_states


class QFormer(nn.Module):
    """
    Q-Former (Querying Transformer) for vision-language bridging.

    Uses learned query tokens to extract visual features from a vision encoder
    that are relevant for language generation.
    """

    def __init__(
        self,
        num_query_token: int = 32,
        vision_width: int = 192,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        cross_attention_freq: int = 2,
    ):
        super().__init__()

        # Create config
        self.config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            vocab_size=30522,  # BERT vocab size
            layer_norm_eps=1e-12,
        )
        # Add custom attributes
        self.config.encoder_width = vision_width
        self.config.add_cross_attention = True
        self.config.cross_attention_freq = cross_attention_freq
        self.config.query_length = num_query_token

        # Embeddings
        self.embeddings = QFormerEmbeddings(self.config)

        # Encoder
        self.encoder = QFormerEncoder(self.config)

        # Query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=0.02)

        self.num_query_token = num_query_token
        self.hidden_size = hidden_size

    def forward(
        self,
        query_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass of Q-Former.

        Args:
            query_embeds: Query embeddings (if None, uses learned query tokens).
            encoder_hidden_states: Vision features from visual encoder.
            encoder_attention_mask: Attention mask for vision features.
            input_ids: Optional text input IDs.
            attention_mask: Optional attention mask for text.

        Returns:
            Encoded query representations.
        """
        if query_embeds is None:
            query_embeds = self.query_tokens

        batch_size = encoder_hidden_states.shape[0] if encoder_hidden_states is not None else query_embeds.shape[0]
        query_embeds = query_embeds.expand(batch_size, -1, -1)

        # Embed input
        embedding_output = self.embeddings(
            input_ids=input_ids,
            query_embeds=query_embeds,
        )

        query_length = query_embeds.shape[1]

        # Create attention masks
        if attention_mask is not None:
            # Extend attention mask for query tokens
            query_attn = torch.ones(
                batch_size, query_length,
                device=attention_mask.device, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([query_attn, attention_mask], dim=1)

        if encoder_attention_mask is not None:
            # Extend encoder attention mask to [B, 1, 1, N]
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
            encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0

        if attention_mask is not None:
            # Extend self attention mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        # Forward through encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            query_length=query_length,
        )

        return encoder_outputs[:, :query_length, :]

    def get_output_dim(self):
        """Get output dimension."""
        return self.hidden_size


def create_qformer(
    num_query_token: int = 32,
    vision_width: int = 192,
    hidden_size: int = 768,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 12,
    cross_attention_freq: int = 2,
    freeze: bool = False,
) -> Tuple[QFormer, nn.Parameter]:
    """
    Create Q-Former with specified configuration.

    Args:
        num_query_token: Number of query tokens.
        vision_width: Width of vision encoder output.
        hidden_size: Hidden size of Q-Former.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        cross_attention_freq: Frequency of cross-attention layers.
        freeze: Whether to freeze the Q-Former.

    Returns:
        qformer: Q-Former model.
        query_tokens: Learned query tokens.
    """
    logging.info(f"Creating Q-Former with {num_query_token} query tokens")

    qformer = QFormer(
        num_query_token=num_query_token,
        vision_width=vision_width,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        cross_attention_freq=cross_attention_freq,
    )

    if freeze:
        for param in qformer.parameters():
            param.requires_grad = False
        qformer.eval()
        logging.info("Froze Q-Former")

    logging.info(f"Q-Former created: {num_hidden_layers} layers, {hidden_size} hidden size")

    return qformer, qformer.query_tokens

