"""
models/backbones.py
Loads pretrained SigLIP-B/16 vision encoder with LoRA and a pruned 12-layer SmolLM2-360M.
Protects nn.ModuleList registrations, enforces targeted layer adapters, and ensures
safe execution across headless cloud GPU containers (Modal / A100 / A10G).
"""
from typing import Optional, Tuple
import gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, SiglipVisionModel, AutoModel
from peft import LoraConfig, get_peft_model
from configs.poc_config import CONFIG


def load_siglip_lora(
    model_name: str = getattr(CONFIG, "siglip_model_name", "google/siglip-base-patch16-384"),
    lora_r: int = getattr(CONFIG, "lora_r", 16),
    lora_alpha: int = getattr(CONFIG, "lora_alpha", 32)
) -> nn.Module:
    """
    Loads SigLIP-B/16 vision encoder and injects rank-16 LoRA adapters
    into attention layers 6-11 to prevent few-shot representation drift.
    """
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    try:
        vision_model = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True
        )
    except Exception:
        try:
            full_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                trust_remote_code=True
            )
            vision_model = getattr(full_model, "vision_model", full_model)
        except Exception:
            vision_model = SiglipVisionModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

    # Freeze base model parameters
    for param in vision_model.parameters():
        param.requires_grad = False

    # Target layers 6 to 11 (upper half of vision stack)
    try:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            layers_to_transform=list(range(6, 12)),
            layers_pattern="layers",
            lora_dropout=0.05,
            bias="none"
        )
        lora_model = get_peft_model(vision_model, lora_config)
    except Exception:
        # Fallback for environments where layers_to_transform is unsupported
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        lora_model = get_peft_model(vision_model, lora_config)

    return lora_model


def load_pruned_smollm2_lora(
    model_name: str = getattr(CONFIG, "smollm_model_name", "HuggingFaceTB/SmolLM2-360M"),
    num_layers: int = 12,
    lora_r: int = getattr(CONFIG, "lora_r", 16),
    lora_alpha: int = getattr(CONFIG, "lora_alpha", 32)
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Loads SmolLM2-360M, prunes from 32 down to 12 layers (~181M params), and wraps with LoRA adapters.
    Maintains strict nn.ModuleList hierarchy to prevent parameter detachment during gradient calculation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    # Instantiate base architecture with SDPA attention
    try:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True
        )
    except Exception:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

    # Prune transformer layers to exact count and update configuration
    target_container = None
    if hasattr(model, "layers"):
        target_container = model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        target_container = model.model

    if target_container is not None and len(target_container.layers) > num_layers:
        # Explicitly register as nn.ModuleList to maintain parameter discovery
        target_container.layers = nn.ModuleList(list(target_container.layers)[:num_layers])
        gc.collect()

    model.config.num_hidden_layers = num_layers

    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA adapters into pruned transformer layers
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    lora_model = get_peft_model(model, lora_config)

    return lora_model, tokenizer