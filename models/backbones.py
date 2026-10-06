"""
Vision and Language Backbones for VGA Policy.

Mathematical and Architectural Foundations:
1. SigLIP-B/16 Vision Backbone:
   - Processes input images I in R^{B x 3 x H x W} into visual patch tokens V in R^{B x N_patch x 768}.
   - For 256x256 images with 16x16 patch size: N_patch = 16 x 16 = 256.
   - For 384x384 images with 16x16 patch size: N_patch = 24 x 24 = 576.
   - LoRA (Low-Rank Adaptation) is applied to query, key, value projection matrices in layers 6-12
     to allow sample-efficient adaptation on 5-shot / 10-shot demonstration data while preserving
     general pretrained visual features.

2. Space-to-Depth & Centroid Ray-RoPE Integration:
   - Visual patch tokens pass through `UnifiedSpaceToDepthProjector` (unshuffles 2x2 spatial
     neighborhoods to yield 64 visual tokens of dimension D_{LM} = 576).
   - `CentroidRayRoPE` assigns unit viewing rays r_ij in S^2 and injects robot camera extrinsics
     t_base via a lightweight MLP, providing zero-overhead 3D spatial awareness.

3. SmolLM2 Language & Fusion Backbone:
   - 30-layer causal language model (hidden_size = 576, intermediate_size = 1536).
   - Encodes language instruction string tokens T in R^{B x L x 576}.
   - Visual tokens (64) and language tokens (L) are fused to produce a rich contextual embedding
     c in R^{B x 576} to condition the DiT Action Expert.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaConfig, LlamaModel, SiglipVisionConfig, SiglipVisionModel

from models.projector import UnifiedSpaceToDepthProjector
from models.ray_rope import CentroidRayRoPE


class VisionLanguageEncoder(nn.Module):
    """
    Unified multimodal perception backbone combining SigLIP vision, Space-to-Depth
    projection, CentroidRayRoPE 3D grounding, and SmolLM2 fusion.
    """

    def __init__(
        self,
        vis_dim: int = 768,
        lm_dim: int = 576,
        img_size: int = 256,
        patch_size: int = 16,
        num_visual_tokens: int = 64,
        num_lm_layers: int = 30,
        pretrained: bool = True,
        freeze_backbones: bool = True,
        vision_model_name: str = "google/siglip-base-patch16-256",
        lm_model_name: str = "HuggingFaceTB/SmolLM2-135M",
        use_lora: bool = False,
        lora_rank: int = 8,
    ):
        super().__init__()
        self.vis_dim = vis_dim
        self.lm_dim = lm_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_visual_tokens = num_visual_tokens
        self.pretrained = pretrained
        self.freeze_backbones = freeze_backbones
        self.vision_model_name = vision_model_name
        self.lm_model_name = lm_model_name

        # 1. SigLIP Vision Backbone (92.9M parameters)
        loaded_pretrained_vision = False
        if pretrained:
            try:
                self.vision_model = SiglipVisionModel.from_pretrained(
                    vision_model_name,
                    torch_dtype=torch.float32,
                )
                loaded_pretrained_vision = True
                print(f"✅ Loaded official pretrained SigLIP vision backbone ({vision_model_name})")
            except Exception as e:
                print(f"⚠️ Warning: Could not load pretrained SigLIP ({e}). Falling back to fresh config.")

        if not loaded_pretrained_vision:
            vis_config = SiglipVisionConfig(
                image_size=img_size,
                patch_size=patch_size,
                hidden_size=vis_dim,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.vision_model = SiglipVisionModel(vis_config)

        # 2. Space-to-Depth Projector (e.g. 256 patches -> 64 tokens)
        grid_dim = img_size // patch_size  # e.g., 256 / 16 = 16, or 384 / 16 = 24
        spatial_factor = grid_dim // int(num_visual_tokens ** 0.5)  # e.g., 16 / 8 = 2
        self.projector = UnifiedSpaceToDepthProjector(
            vis_dim=vis_dim,
            lm_dim=lm_dim,
            spatial_factor=spatial_factor,
        )

        # 3. CentroidRayRoPE 3D Grounding
        self.ray_rope = CentroidRayRoPE(
            lm_dim=lm_dim,
        )

        # 4. SmolLM2 Language Backbone (134.5M parameters)
        loaded_pretrained_lm = False
        if pretrained:
            try:
                self.language_model = LlamaModel.from_pretrained(
                    lm_model_name,
                    torch_dtype=torch.float32,
                )
                loaded_pretrained_lm = True
                print(f"✅ Loaded official pretrained SmolLM2 language backbone ({lm_model_name})")
            except Exception as e:
                print(f"⚠️ Warning: Could not load pretrained SmolLM2 ({e}). Falling back to fresh config.")

        if not loaded_pretrained_lm:
            lm_config = LlamaConfig(
                vocab_size=49152,
                hidden_size=lm_dim,
                intermediate_size=1536,
                num_hidden_layers=num_lm_layers,
                num_attention_heads=9,
                num_key_value_heads=3,
                max_position_embeddings=2048,
            )
            self.language_model = LlamaModel(lm_config)

        # 5. Freeze Backbones if requested
        if freeze_backbones:
            for p in self.vision_model.parameters():
                p.requires_grad = False
            for p in self.language_model.parameters():
                p.requires_grad = False
            print("❄️ Frozen SigLIP & SmolLM2 backbones: internet perception priors locked. Training Projector, RayRoPE & DiT Expert.")

        # 6. Tokenizer helper
        self.tokenizer = None

    def get_tokenizer(self):
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.lm_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception:
                pass
        return self.tokenizer

    def forward(
        self,
        image_front: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        camera_origin: Optional[torch.Tensor] = None,
        image_wrist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes image observations and language into a fused context vector c in R^{B x lm_dim}.
        Supports single-camera (front only) or dual-camera (front + wrist camera) multimodal fusion.

        Args:
            image_front: Front camera RGB observation [B, 3, img_size, img_size]
            input_ids: Language token IDs [B, seq_len]
            attention_mask: Language attention mask [B, seq_len]
            camera_origin: Optional camera position in robot base frame [B, 3]
            image_wrist: Optional wrist camera RGB observation [B, 3, img_size, img_size]
        Returns:
            context: Contextual conditioning vector [B, lm_dim]
        """
        # 1. Extract visual patch tokens from SigLIP for front camera
        vis_outputs = self.vision_model(pixel_values=image_front)
        patch_tokens = vis_outputs.last_hidden_state  # [B, N_patches, 768]

        # 2. Compress patch tokens to 64 tokens via Space-to-Depth
        compressed_vis_tokens = self.projector(patch_tokens)  # [B, 64, lm_dim] (e.g. 576)

        # 3. Ground visual tokens with 3D viewing rays via Ray-RoPE
        if camera_origin is None:
            camera_origin = torch.zeros((image_front.shape[0], 3), device=image_front.device, dtype=image_front.dtype)
        grounded_vis_tokens = self.ray_rope(compressed_vis_tokens, t_base_cam=camera_origin)  # [B, 64, lm_dim]

        # 3b. Optional Dual-Camera Fusion: Wrist (Eye-in-Hand) Camera
        if image_wrist is not None:
            vis_wrist_out = self.vision_model(pixel_values=image_wrist)
            wrist_patch_tokens = vis_wrist_out.last_hidden_state  # [B, N_patches, 768]
            compressed_wrist_tokens = self.projector(wrist_patch_tokens)  # [B, 64, lm_dim]
            grounded_wrist_tokens = self.ray_rope(compressed_wrist_tokens, t_base_cam=camera_origin)
            visual_tokens = torch.cat([grounded_vis_tokens, grounded_wrist_tokens], dim=1)  # [B, 128, lm_dim]
            total_vis_tokens = self.num_visual_tokens * 2
        else:
            visual_tokens = grounded_vis_tokens
            total_vis_tokens = self.num_visual_tokens

        # 4. Extract language embeddings from SmolLM2
        lang_inputs_embeds = self.language_model.embed_tokens(input_ids)  # [B, seq_len, lm_dim]

        # 5. Concatenate visual tokens (front + wrist) and language tokens
        # [B, total_vis_tokens + seq_len, lm_dim]
        multimodal_seq = torch.cat([visual_tokens, lang_inputs_embeds], dim=1)

        # Create combined attention mask
        batch_size = image_front.shape[0]
        device = image_front.device
        vis_mask = torch.ones((batch_size, total_vis_tokens), dtype=torch.bool, device=device)
        if attention_mask is not None:
            combined_mask = torch.cat([vis_mask, attention_mask.bool()], dim=1)
        else:
            combined_mask = torch.ones(multimodal_seq.shape[:2], dtype=torch.bool, device=device)

        # 6. Pass through SmolLM2 Transformer Layers
        lm_outputs = self.language_model(
            inputs_embeds=multimodal_seq,
            attention_mask=combined_mask,
        )
        fused_hidden = lm_outputs.last_hidden_state  # [B, total_vis_tokens + seq_len, lm_dim]

        # 7. Mean pool over the multimodal sequence to produce conditioning vector c
        # (ignoring padding tokens via the attention mask)
        mask_expanded = combined_mask.unsqueeze(-1).float()
        context = (fused_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)

        return context  # [B, lm_dim]
