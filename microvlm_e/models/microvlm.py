"""
MicroVLM: Main Vision-Language Model for MicroVLM-E.

Combines:
- DiET vision encoder (frozen)
- Q-Former for vision-language bridging
- Two projection layers
- Qwen2.5-0.5B language model with LoRA/QLoRA adapters
"""

import logging
import random
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList

from microvlm_e.common.registry import registry
from microvlm_e.models.base_model import BaseModel, LayerNorm, RMSNorm, disabled_train
from microvlm_e.models.diet_encoder import create_diet_encoder
from microvlm_e.models.qformer import QFormer, create_qformer


class StoppingCriteriaSub(StoppingCriteria):
    """Custom stopping criteria for text generation."""

    def __init__(self, stops: List[torch.Tensor] = None, encounters: int = 1):
        super().__init__()
        self.stops = stops if stops is not None else []
        self.encounters = encounters

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


@registry.register_model("microvlm")
class MicroVLM(BaseModel):
    """
    MicroVLM: Micro Vision-Language Model.

    Architecture:
    1. Frozen DiET vision encoder
    2. Q-Former with learned query tokens
    3. Two projection layers
    4. Qwen2.5-0.5B with LoRA/QLoRA adapters
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "default": "configs/models/microvlm_default.yaml",
    }

    def __init__(
        self,
        # Vision encoder
        vision_encoder: str = "diet_tiny",
        img_size: int = 224,
        drop_path_rate: float = 0.0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,

        # Q-Former
        num_query_token: int = 32,
        qformer_hidden_size: int = 768,
        qformer_num_layers: int = 6,
        cross_attention_freq: int = 2,
        freeze_qformer: bool = False,

        # Language model
        llm_model: str = "Qwen/Qwen2.5-0.5B",
        max_txt_len: int = 512,
        max_context_len: int = 2048,
        prompt_template: str = "",
        end_sym: str = "\n",

        # LoRA settings
        use_lora: bool = True,
        use_qlora: bool = False,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,

        # Quantization settings
        low_resource: bool = False,
        device_8bit: int = 0,
    ):
        super().__init__()

        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym
        self.prompt_template = prompt_template
        self.low_resource = low_resource
        self.use_qlora = use_qlora

        # ========== Vision Encoder ==========
        logging.info(f"Initializing vision encoder: {vision_encoder}")
        self.visual_encoder, self.ln_vision = create_diet_encoder(
            model_name=vision_encoder,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            pretrained=True,
            freeze=freeze_vit,
            qk_norm=True,
        )
        vision_width = self.visual_encoder.embed_dim

        # ========== Q-Former ==========
        logging.info("Initializing Q-Former")
        self.Qformer, self.query_tokens = create_qformer(
            num_query_token=num_query_token,
            vision_width=vision_width,
            hidden_size=qformer_hidden_size,
            num_hidden_layers=qformer_num_layers,
            cross_attention_freq=cross_attention_freq,
            freeze=freeze_qformer,
        )
        qformer_output_dim = self.Qformer.hidden_size

        # ========== Language Model ==========
        logging.info(f"Initializing LLM: {llm_model}")
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.llm_model, self.llm_tokenizer = self.init_llm(
            llm_model_path=llm_model,
            low_resource=low_resource,
            low_res_device=device_8bit,
            use_qlora=use_qlora,
            lora_r=lora_r if use_lora else 0,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Get LLM hidden size
        self.llm_hidden_size = self.llm_model.config.hidden_size

        # ========== Projection Layers ==========
        # Two projection layers: Q-Former output -> intermediate -> LLM hidden size
        intermediate_size = (qformer_output_dim + self.llm_hidden_size) // 2

        logging.info(f"Initializing projection layers: {qformer_output_dim} -> {intermediate_size} -> {self.llm_hidden_size}")

        self.llm_proj1 = nn.Linear(qformer_output_dim, intermediate_size)
        self.llm_proj2 = nn.Linear(intermediate_size, self.llm_hidden_size)

        # Additional normalization
        self.proj_norm = LayerNorm(intermediate_size)

        # ========== Print Summary ==========
        logging.info(f"\n{'='*50}")
        logging.info("MicroVLM Architecture Summary:")
        logging.info(f"  Vision Encoder: {vision_encoder} (dim={vision_width})")
        logging.info(f"  Q-Former: {qformer_num_layers} layers, {num_query_token} queries, dim={qformer_output_dim}")
        logging.info(f"  Projection: {qformer_output_dim} -> {intermediate_size} -> {self.llm_hidden_size}")
        logging.info(f"  LLM: {llm_model} (dim={self.llm_hidden_size})")
        logging.info(f"  LoRA: {'Enabled' if use_lora else 'Disabled'} (r={lora_r})")
        logging.info(f"  QLoRA: {'Enabled' if use_qlora else 'Disabled'}")
        logging.info(f"{'='*50}")
        logging.info(self.show_n_params())

    def vit_to_cpu(self):
        """Move vision encoder to CPU (for low memory mode)."""
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from LLM."""
        if hasattr(self.llm_model, "model"):
            # For PEFT wrapped models
            if hasattr(self.llm_model.model, "model"):
                embed = self.llm_model.model.model.embed_tokens
            else:
                embed = self.llm_model.model.embed_tokens
        else:
            embed = self.llm_model.embed_tokens

        return embed(token_ids)

    def encode_img(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image through vision encoder and Q-Former.

        Args:
            image: Input image tensor of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            inputs_llm: Projected visual features for LLM input.
            atts_llm: Attention mask for visual features.
        """
        device = image.device

        # Handle video input (B, T, C, H, W)
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            # Vision encoder
            image_embeds = self.visual_encoder(image)  # (B, N, D)
            image_embeds = self.ln_vision(image_embeds).to(device)

            # Attention mask for image embeddings
            image_atts = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=device
            )

            # Q-Former
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
            )

            # Projection layers
            proj_features = self.llm_proj1(query_output)
            proj_features = F.gelu(proj_features)
            proj_features = self.proj_norm(proj_features)
            inputs_llm = self.llm_proj2(proj_features)

            atts_llm = torch.ones(
                inputs_llm.size()[:-1],
                dtype=torch.long,
                device=device
            )

        return inputs_llm, atts_llm

    def get_context_emb(
        self,
        prompt: str,
        img_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Create context embedding by interleaving text and image embeddings.

        Args:
            prompt: Text prompt with <ImageHere> placeholders.
            img_list: List of image embeddings.

        Returns:
            Mixed embeddings tensor.
        """
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')

        assert len(prompt_segs) == len(img_list) + 1, \
            f"Unmatched numbers of image placeholders ({len(prompt_segs)-1}) and images ({len(img_list)})"

        seg_tokens = [
            self.llm_tokenizer(
                seg,
                return_tensors="pt",
                add_special_tokens=(i == 0)
            ).to(device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]

        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        # Interleave embeddings
        mixed_embs = []
        for i, (seg_emb, img_emb) in enumerate(zip(seg_embs[:-1], img_list)):
            mixed_embs.append(seg_emb)
            mixed_embs.append(img_emb)
        mixed_embs.append(seg_embs[-1])

        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def prompt_wrap(
        self,
        img_embeds: torch.Tensor,
        atts_img: torch.Tensor,
        prompts: Optional[List[str]],
        lengths: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrap image embeddings with text prompts.

        Args:
            img_embeds: Image embeddings from encode_img.
            atts_img: Attention mask for image embeddings.
            prompts: Text prompts (can contain <ImageHere>).
            lengths: Optional lengths for multi-image input.

        Returns:
            wrapped_embs: Combined embeddings.
            wrapped_atts: Combined attention mask.
        """
        if prompts is None or len(prompts) == 0:
            return img_embeds, atts_img

        if img_embeds is None:
            # Text-only input
            self.llm_tokenizer.padding_side = "right"
            prompt_tokens = self.llm_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt

        # Multi-modal input
        emb_lists = []
        if isinstance(prompts, str):
            prompts = [prompts] * len(img_embeds)

        for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
            pn = each_img_embed.shape[-2]

            if lengths is not None:
                each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                each_img_embed = each_img_embed[:lengths[idx] * pn]

            p_segs = each_prompt.split('<ImageHere>')
            interleave_emb = []

            for i, seg in enumerate(p_segs[:-1]):
                p_tokens = self.llm_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=False
                ).to(img_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                interleave_emb.append(
                    torch.cat([p_embed, each_img_embed[None][:, i*pn:(i+1)*pn]], dim=1)
                )

            wrapped_emb = torch.cat(interleave_emb, dim=1)

            # Add final text segment
            p_tokens = self.llm_tokenizer(
                p_segs[-1], return_tensors="pt", add_special_tokens=False
            ).to(img_embeds.device)
            p_embed = self.embed_tokens(p_tokens.input_ids)
            wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)

            emb_lists.append(wrapped_emb)

        # Pad to same length
        emb_lens = [emb.shape[1] for emb in emb_lists]
        max_length = min(max(emb_lens), self.max_context_len)

        pad_emb = self.embed_tokens(
            torch.tensor([self.llm_tokenizer.pad_token_id], device=img_embeds.device)
        )

        wrapped_embs = pad_emb.expand(len(emb_lists), max_length, -1).clone()
        wrapped_atts = torch.zeros(
            [len(emb_lists), max_length],
            dtype=torch.int,
            device=img_embeds.device
        )

        for i, emb in enumerate(emb_lists):
            length = min(emb_lens[i], max_length)
            wrapped_embs[i, :length] = emb[:, :length]
            wrapped_atts[i, :length] = 1

        return wrapped_embs, wrapped_atts

    def forward(
        self,
        samples: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            samples: Dictionary containing:
                - image: Input images (B, C, H, W)
                - text_input: Input text prompts
                - text_output: Target text outputs (optional)

        Returns:
            Dictionary containing loss and other metrics.
        """
        image = samples.get("image")
        text_input = samples.get("text_input", samples.get("prompt", [""]))
        text_output = samples.get("text_output", samples.get("answer", None))

        # Encode image
        if image is not None:
            img_embeds, atts_img = self.encode_img(image)
        else:
            img_embeds, atts_img = None, None

        # Prepare prompts
        if self.prompt_template:
            prompts = [self.prompt_template.format(t) for t in text_input]
        else:
            prompts = text_input

        # Add image placeholder if not present
        if img_embeds is not None:
            prompts = [
                p if "<ImageHere>" in p else "<ImageHere> " + p
                for p in prompts
            ]

        # Wrap prompts with image embeddings
        wrapped_embs, wrapped_atts = self.prompt_wrap(img_embeds, atts_img, prompts)

        # Tokenize targets
        if text_output is not None:
            self.llm_tokenizer.padding_side = "right"
            targets_tokens = self.llm_tokenizer(
                [t + self.end_sym for t in text_output],
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False,
                max_length=self.max_txt_len,
                truncation=True,
            ).to(wrapped_embs.device)

            target_embeds = self.embed_tokens(targets_tokens.input_ids)
            target_atts = targets_tokens.attention_mask

            # Concatenate input and target
            inputs_embeds = torch.cat([wrapped_embs, target_embeds], dim=1)
            attention_mask = torch.cat([wrapped_atts, target_atts], dim=1)

            # Create labels (mask input portion)
            targets = targets_tokens.input_ids.masked_fill(
                targets_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
            )
            empty_targets = torch.ones(
                [wrapped_atts.shape[0], wrapped_atts.shape[1]],
                dtype=torch.long,
                device=wrapped_embs.device
            ).fill_(-100)
            targets = torch.cat([empty_targets, targets], dim=1)

            # Forward through LLM
            with self.maybe_autocast():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

            loss = outputs.loss

            return {
                "loss": loss,
            }
        else:
            # Inference mode
            return {
                "inputs_embeds": wrapped_embs,
                "attention_mask": wrapped_atts,
            }

    @torch.no_grad()
    def generate(
        self,
        samples: Dict[str, Any],
        max_new_tokens: int = 128,
        min_length: int = 1,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        stop_words: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate text from image and prompt.

        Args:
            samples: Dictionary containing image and prompt.
            max_new_tokens: Maximum number of tokens to generate.
            min_length: Minimum generation length.
            top_p: Top-p sampling parameter.
            temperature: Sampling temperature.
            repetition_penalty: Repetition penalty.
            do_sample: Whether to use sampling.
            num_beams: Number of beams for beam search.
            stop_words: List of stop words.

        Returns:
            List of generated text strings.
        """
        image = samples.get("image")
        prompt = samples.get("prompt", samples.get("text_input", ""))

        if isinstance(prompt, str):
            prompt = [prompt]

        # Encode image
        if image is not None:
            img_embeds, atts_img = self.encode_img(image)
        else:
            img_embeds, atts_img = None, None

        # Prepare prompts
        if self.prompt_template:
            prompts = [self.prompt_template.format(p) for p in prompt]
        else:
            prompts = prompt

        # Add image placeholder if needed
        if img_embeds is not None:
            prompts = [
                p if "<ImageHere>" in p else "<ImageHere> " + p
                for p in prompts
            ]

        # Wrap prompts
        inputs_embeds, attention_mask = self.prompt_wrap(img_embeds, atts_img, prompts)

        # Setup stopping criteria
        stopping_criteria = StoppingCriteriaList()
        if stop_words:
            stop_ids = [
                self.llm_tokenizer(w, return_tensors="pt", add_special_tokens=False).input_ids[0]
                for w in stop_words
            ]
            stopping_criteria.append(StoppingCriteriaSub(stops=stop_ids))

        # Generate
        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )

        # Decode outputs
        output_texts = self.llm_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return output_texts

    @classmethod
    def from_config(cls, cfg) -> "MicroVLM":
        """Create model from configuration."""
        return cls(
            # Vision encoder
            vision_encoder=cfg.get("vision_encoder", "diet_tiny"),
            img_size=cfg.get("image_size", 224),
            drop_path_rate=cfg.get("drop_path_rate", 0.0),
            use_grad_checkpoint=cfg.get("use_grad_checkpoint", False),
            vit_precision=cfg.get("vit_precision", "fp16"),
            freeze_vit=cfg.get("freeze_vit", True),

            # Q-Former
            num_query_token=cfg.get("num_query_token", 32),
            qformer_hidden_size=cfg.get("qformer_hidden_size", 768),
            qformer_num_layers=cfg.get("qformer_num_layers", 6),
            cross_attention_freq=cfg.get("cross_attention_freq", 2),
            freeze_qformer=cfg.get("freeze_qformer", False),

            # LLM
            llm_model=cfg.get("llm_model", "Qwen/Qwen2.5-0.5B"),
            max_txt_len=cfg.get("max_txt_len", 512),
            max_context_len=cfg.get("max_context_len", 2048),
            prompt_template=cfg.get("prompt_template", ""),
            end_sym=cfg.get("end_sym", "\n"),

            # LoRA
            use_lora=cfg.get("use_lora", True),
            use_qlora=cfg.get("use_qlora", False),
            lora_r=cfg.get("lora_r", 64),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            lora_target_modules=cfg.get("lora_target_modules", None),

            # Resource
            low_resource=cfg.get("low_resource", False),
            device_8bit=cfg.get("device_8bit", 0),
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = []

        # Q-Former parameters (if not frozen)
        for p in self.Qformer.parameters():
            if p.requires_grad:
                params.append(p)

        # Query tokens
        if self.query_tokens.requires_grad:
            params.append(self.query_tokens)

        # Projection layers
        for p in self.llm_proj1.parameters():
            params.append(p)
        for p in self.llm_proj2.parameters():
            params.append(p)
        for p in self.proj_norm.parameters():
            params.append(p)

        # LoRA parameters in LLM
        for p in self.llm_model.parameters():
            if p.requires_grad:
                params.append(p)

        return params

