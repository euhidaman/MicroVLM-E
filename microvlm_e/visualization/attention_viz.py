"""
Attention visualization module for MicroVLM-E.

Provides tools for visualizing cross-attention between image and text tokens,
generating heatmaps showing which image regions the model attends to.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class AttentionExtractor:
    """
    Extracts attention maps from MicroVLM model components.

    Supports extraction from:
    - Q-Former cross-attention
    - Projection layers (via gradient-based attention)
    - LLM multimodal input embeddings
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to capture attention maps."""
        self.clear_hooks()

        # Hook for Q-Former attention
        if hasattr(self.model, 'Qformer'):
            for name, module in self.model.Qformer.named_modules():
                if 'crossattention' in name.lower() and hasattr(module, 'self'):
                    hook = module.self.register_forward_hook(
                        self._create_attention_hook(f"qformer.{name}")
                    )
                    self.hooks.append(hook)

        # Hook for LLM attention layers
        if hasattr(self.model, 'llm_model'):
            for name, module in self.model.llm_model.named_modules():
                if 'attention' in name.lower() and hasattr(module, 'forward'):
                    if 'self_attn' in name or 'attention' in name:
                        hook = module.register_forward_hook(
                            self._create_attention_hook(f"llm.{name}")
                        )
                        self.hooks.append(hook)

    def _create_attention_hook(self, name: str):
        """Create a hook function that captures attention weights."""
        def hook(module, input, output):
            # Try to extract attention weights from output
            if isinstance(output, tuple) and len(output) > 1:
                # Some attention modules return (output, attention_weights)
                if output[1] is not None and isinstance(output[1], torch.Tensor):
                    self.attention_maps[name] = output[1].detach().cpu()
            elif hasattr(module, 'attention_probs'):
                self.attention_maps[name] = module.attention_probs.detach().cpu()
        return hook

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = {}

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Get all captured attention maps."""
        return self.attention_maps

    def __del__(self):
        self.clear_hooks()


class AttentionVisualizer:
    """
    Visualizes attention maps as heatmaps overlaid on images.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 224,
        patch_size: int = 16,
        colormap: str = 'jet',
    ):
        self.model = model
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.colormap = colormap
        self.extractor = AttentionExtractor(model)

    def compute_attention_rollout(
        self,
        attention_maps: List[torch.Tensor],
        discard_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute attention rollout across multiple layers.

        Args:
            attention_maps: List of attention tensors [B, H, N, N]
            discard_ratio: Ratio of lowest attention values to discard

        Returns:
            Rolled out attention map
        """
        result = torch.eye(attention_maps[0].shape[-1])

        for attention in attention_maps:
            # Average over heads
            attention = attention.mean(dim=1)

            # Add residual connection
            attention = attention + torch.eye(attention.shape[-1])

            # Normalize
            attention = attention / attention.sum(dim=-1, keepdim=True)

            # Discard lowest values
            if discard_ratio > 0:
                flat = attention.view(-1)
                threshold = flat.kthvalue(int(flat.numel() * discard_ratio)).values
                attention = torch.where(attention > threshold, attention, torch.zeros_like(attention))

            result = torch.matmul(attention, result)

        return result

    def get_qformer_attention(
        self,
        image: torch.Tensor,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get Q-Former cross-attention maps.

        Args:
            image: Input image tensor [B, C, H, W]
            return_all_layers: Whether to return attention from all layers

        Returns:
            Attention map(s) showing query-to-image-patch attention
        """
        self.extractor.register_hooks()

        with torch.no_grad():
            # Encode image through Q-Former
            self.model.encode_img(image)

        attention_maps = self.extractor.get_attention_maps()
        self.extractor.clear_hooks()

        # Filter for Q-Former attention maps
        qformer_attns = [v for k, v in attention_maps.items() if 'qformer' in k]

        if return_all_layers:
            return qformer_attns
        elif qformer_attns:
            # Return average across layers
            return torch.stack(qformer_attns).mean(dim=0)
        else:
            return None

    def generate_heatmap(
        self,
        attention: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate a heatmap from attention weights.

        Args:
            attention: Attention tensor [N_queries, N_patches] or similar
            image_size: Target image size (H, W)

        Returns:
            Heatmap as numpy array [H, W, 3]
        """
        if image_size is None:
            image_size = (self.image_size, self.image_size)

        # Handle different attention shapes
        if attention.dim() == 4:  # [B, H, Q, K]
            attention = attention.mean(dim=1).mean(dim=0)  # Average heads and batch
        elif attention.dim() == 3:  # [B, Q, K] or [H, Q, K]
            attention = attention.mean(dim=0)

        # If attention is query-to-patch, average over queries
        if attention.shape[0] < attention.shape[1]:
            # Queries attending to patches
            attention = attention.mean(dim=0)
        elif attention.shape[0] > attention.shape[1]:
            # Patches attending to queries (transpose)
            attention = attention.mean(dim=1)

        # Compute grid size
        grid_size = int(np.sqrt(len(attention)))
        if grid_size ** 2 != len(attention):
            # Handle CLS token or other extras
            grid_size = int(np.sqrt(len(attention) - 1))
            attention = attention[1:]  # Remove CLS

        # Reshape to grid
        attention = attention[:grid_size**2].reshape(grid_size, grid_size)

        # Normalize
        attention = attention.numpy() if isinstance(attention, torch.Tensor) else attention
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        # Upsample to image size
        attention_upsampled = np.array(Image.fromarray(
            (attention * 255).astype(np.uint8)
        ).resize(image_size, Image.BILINEAR)) / 255.0

        # Apply colormap
        cmap = cm.get_cmap(self.colormap)
        heatmap = cmap(attention_upsampled)[:, :, :3]

        return (heatmap * 255).astype(np.uint8)

    def overlay_heatmap(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image
            heatmap: Heatmap array [H, W, 3]
            alpha: Blend factor (0=image only, 1=heatmap only)

        Returns:
            Blended image as numpy array
        """
        # Convert image to numpy array
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Resize heatmap to match image
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = np.array(Image.fromarray(heatmap).resize(
                (image.shape[1], image.shape[0]), Image.BILINEAR
            ))

        # Blend
        overlay = (1 - alpha) * image + alpha * heatmap
        return overlay.astype(np.uint8)

    def visualize_token_attention(
        self,
        image: Union[Image.Image, torch.Tensor],
        attention: torch.Tensor,
        tokens: List[str],
        output_dir: str,
        prefix: str = "token_attn",
    ) -> List[str]:
        """
        Generate per-token attention visualizations.

        Args:
            image: Input image
            attention: Attention tensor [N_tokens, N_patches]
            tokens: List of token strings
            output_dir: Directory to save visualizations
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, token in enumerate(tokens):
            if i >= attention.shape[0]:
                break

            token_attn = attention[i]
            heatmap = self.generate_heatmap(token_attn)
            overlay = self.overlay_heatmap(image, heatmap, alpha=0.6)

            # Clean token for filename
            clean_token = token.replace("/", "_").replace("\\", "_")[:20]
            filename = f"{prefix}_{i:03d}_{clean_token}.png"
            filepath = os.path.join(output_dir, filename)

            # Create figure with image and token label
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(overlay)
            ax.set_title(f"Token: '{token}'", fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            saved_paths.append(filepath)

        return saved_paths


class AlignmentVisualizer:
    """
    High-level API for visualizing image-text alignment in MicroVLM.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer or model.llm_tokenizer
        self.device = device
        self.visualizer = AttentionVisualizer(model)

        # Image preprocessing
        from microvlm_e.processors import ImageEvalProcessor
        self.image_processor = ImageEvalProcessor(image_size=224)

    def visualize(
        self,
        image_path: str,
        text: str,
        output_path: str,
        visualization_type: str = "overlay",
        layer_index: int = -1,
    ) -> str:
        """
        Generate visualization of image-text alignment.

        Args:
            image_path: Path to input image
            text: Text prompt or description
            output_path: Path to save visualization
            visualization_type: Type of visualization ('overlay', 'heatmap', 'tokens')
            layer_index: Which layer's attention to visualize (-1 for average)

        Returns:
            Path to saved visualization
        """
        # Load and process image
        original_image = Image.open(image_path).convert('RGB')
        processed_image = self.image_processor(original_image)
        image_tensor = processed_image.unsqueeze(0).to(self.device)

        # Get attention maps
        attention_maps = self.visualizer.get_qformer_attention(
            image_tensor,
            return_all_layers=True
        )

        if not attention_maps:
            logging.warning("No attention maps captured. Using gradient-based attention.")
            attention = self._compute_gradient_attention(image_tensor, text)
        else:
            if layer_index == -1:
                # Average all layers
                attention = torch.stack(attention_maps).mean(dim=0)
            else:
                attention = attention_maps[layer_index]

        # Generate visualization based on type
        if visualization_type == "heatmap":
            heatmap = self.visualizer.generate_heatmap(
                attention,
                image_size=original_image.size[::-1]
            )
            output_image = Image.fromarray(heatmap)

        elif visualization_type == "overlay":
            heatmap = self.visualizer.generate_heatmap(
                attention,
                image_size=original_image.size[::-1]
            )
            overlay = self.visualizer.overlay_heatmap(original_image, heatmap, alpha=0.5)
            output_image = Image.fromarray(overlay)

        elif visualization_type == "tokens":
            # Tokenize text
            tokens = self.tokenizer.tokenize(text)
            output_dir = os.path.dirname(output_path)
            prefix = os.path.splitext(os.path.basename(output_path))[0]

            saved_paths = self.visualizer.visualize_token_attention(
                original_image,
                attention,
                tokens,
                output_dir,
                prefix
            )
            return saved_paths[0] if saved_paths else output_path

        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")

        # Save output
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        output_image.save(output_path)
        logging.info(f"Visualization saved to: {output_path}")

        return output_path

    def _compute_gradient_attention(
        self,
        image: torch.Tensor,
        text: str,
    ) -> torch.Tensor:
        """
        Compute attention using gradient-based method.

        When hook-based attention extraction fails, use gradients
        to approximate which image regions are important.
        """
        self.model.eval()
        image.requires_grad_(True)

        # Encode image
        img_embeds, _ = self.model.encode_img(image)

        # Tokenize text
        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get text embeddings
        text_embeds = self.model.embed_tokens(text_tokens.input_ids)

        # Compute similarity
        similarity = torch.einsum('bqd,btd->bqt', img_embeds, text_embeds)

        # Backpropagate
        similarity.sum().backward()

        # Get gradient-based attention
        grad_attention = image.grad.abs().mean(dim=1)  # [B, H, W]

        # Convert to patch-level attention
        patch_size = self.visualizer.patch_size
        patches_h = grad_attention.shape[1] // patch_size
        patches_w = grad_attention.shape[2] // patch_size

        patch_attention = grad_attention.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patch_attention = patch_attention.mean(dim=(-1, -2))  # [B, Ph, Pw]
        patch_attention = patch_attention.reshape(-1)  # [Ph * Pw]

        return patch_attention

    def create_attention_video(
        self,
        image_path: str,
        text: str,
        output_path: str,
        fps: int = 2,
    ) -> str:
        """
        Create a video/GIF showing attention evolving across layers.

        Args:
            image_path: Path to input image
            text: Text prompt
            output_path: Path to save video (use .gif extension)
            fps: Frames per second

        Returns:
            Path to saved video
        """
        original_image = Image.open(image_path).convert('RGB')
        processed_image = self.image_processor(original_image)
        image_tensor = processed_image.unsqueeze(0).to(self.device)

        # Get attention from all layers
        attention_maps = self.visualizer.get_qformer_attention(
            image_tensor,
            return_all_layers=True
        )

        if not attention_maps:
            logging.warning("No attention maps available for video creation")
            return output_path

        frames = []
        for i, attn in enumerate(attention_maps):
            heatmap = self.visualizer.generate_heatmap(
                attn,
                image_size=original_image.size[::-1]
            )
            overlay = self.visualizer.overlay_heatmap(original_image, heatmap, alpha=0.5)

            # Add layer label
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(overlay)
            ax.set_title(f"Layer {i+1}/{len(attention_maps)}", fontsize=14)
            ax.axis('off')

            # Convert figure to image
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(Image.fromarray(frame))
            plt.close()

        # Save as GIF
        if frames:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0
            )
            logging.info(f"Attention video saved to: {output_path}")

        return output_path

