from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import os
import pathlib

import torch
import transformers
import yaml  # type: ignore

from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)
from llava.model.language_model.llava_qwen_pointer import (
    LlavaQwen2ForCausalLMWithPointer,
)
from llava.model.llava_arch import LlavaMetaModel
from llava.model import builder as model_builder
from llava.train.llava_trainer import LLaVATrainer
from llava.train.dataset_pointer import PointerSupervisedDataset, PointerCollator, PointerDataArgs
from llava import conversation as conversation_lib


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="qwen_v2")
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_projector_type: Optional[str] = field(default="linear")
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    # Optionally load a pre-trained MLP projector checkpoint path (weights for mm_projector)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_mlp_and_vision_tower: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    s2: Optional[bool] = field(default=False)
    hd: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(default=None)
    image_folder: Optional[List[str]] = field(default=None)
    image_aspect_ratio: str = field(default="square")
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(default=4096)
    # Optional per-module learning rates
    mm_projector_lr: Optional[float] = field(default=None)
    mm_vision_tower_lr: Optional[float] = field(default=None)
    pointer_loss_weight: float = field(default=1.0)
    # Default to pointer-only warmup (GUI-Actor style). Set to 1.0 for SFT.
    lm_loss_weight: float = field(default=0.0)
    # Default to freezing the backbone in warmup. Set to False for SFT.
    freeze_backbone: bool = field(default=True)


def smart_add_special_tokens(tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    # Add image and pointer tokens
    add_dict = {
        "additional_special_tokens": [
            DEFAULT_IMAGE_PATCH_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_POINTER_START_TOKEN,
            DEFAULT_POINTER_END_TOKEN,
            DEFAULT_POINTER_PAD_TOKEN,
        ]
    }
    num_new = tokenizer.add_special_tokens(add_dict)
    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))
        # average-init
        input_emb = model.get_input_embeddings().weight.data
        output_emb = model.get_output_embeddings().weight.data
        input_emb[-num_new:] = input_emb[:-num_new].mean(dim=0, keepdim=True)
        output_emb[-num_new:] = output_emb[:-num_new].mean(dim=0, keepdim=True)

    # Stash pointer IDs on config for model to read
    model.config.pointer_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
    model.config.pointer_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)
    model.config.pointer_pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)


def make_data_module(tokenizer, vision_tower, data_args: DataArguments):
    pdata_args = PointerDataArgs(
        image_folder=data_args.image_folder,
        is_multimodal=True,
        image_aspect_ratio=data_args.image_aspect_ratio,
        mm_use_im_start_end=False,
        image_processor=vision_tower.image_processor,
        image_grid_pinpoints=data_args.image_grid_pinpoints,
    )
    dataset = PointerSupervisedDataset(data_args.data_path, tokenizer, pdata_args)
    collator = PointerCollator(tokenizer)
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": collator}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Normalize/expand data paths: support YAML config from GUI-Actor
    def _parse_yaml_config(path: str) -> Tuple[List[str], List[str]]:
        """Return (json_paths, image_folders) from a YAML config with `datasets:` list.
        Requires PyYAML and a GUI-Actor style schema.
        """
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)  # type: ignore
        ds = cfg.get("datasets", []) if isinstance(cfg, dict) else []
        jsons: List[str] = []
        folders: List[str] = []
        for d in ds:
            if not isinstance(d, dict):
                continue
            jp = d.get("json_path")
            imf = d.get("images_folder")
            if jp is not None:
                jsons.append(str(jp))
                folders.append(str(imf) if imf is not None else "")
        return jsons, folders

    # Expand data_args.data_path from YAML if needed
    if data_args.data_path is None:
        raise ValueError("--data_path must be provided and point to a YAML config")

    # Require a single YAML file (GUI-Actor style)
    yaml_path = data_args.data_path[0] if isinstance(data_args.data_path, list) else data_args.data_path  # type: ignore[index]
    if not isinstance(yaml_path, str) or not yaml_path.endswith((".yaml", ".yml")):
        raise ValueError("--data_path must be a YAML file (GUI-Actor/data/data_config.yaml)")
    jsons, folders = _parse_yaml_config(yaml_path)
    if not jsons:
        raise ValueError(f"No datasets found in YAML: {yaml_path}")
    data_args.data_path = jsons
    data_args.image_folder = folders

    # Guard: avoid 'spatial_unpad' with AnyRes, prefer 'spatial' to match pointer path
    if data_args.image_aspect_ratio == "anyres" and isinstance(model_args.mm_patch_merge_type, str):
        if "unpad" in model_args.mm_patch_merge_type:
            print("[warn] image_aspect_ratio='anyres' with mm_patch_merge_type containing 'unpad' can misalign pointer supervision. Overriding to 'spatial'.")
            model_args.mm_patch_merge_type = "spatial"

    # Load tokenizer and base LLaVA config/model (Qwen2 backbone)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Set conversation template (Qwen2)
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    elif model_args.version.startswith("qwen_v2") or model_args.version.startswith("qwen_2"):
        # Our conv templates register the Qwen2 style under key "qwen_2"
        conversation_lib.default_conversation = conversation_lib.conv_templates["qwen_2"]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # Load first without dtype/device to ensure newly added modules (e.g. pointer head)
    # initialize with real tensors, then cast explicitly.
    model = LlavaQwen2ForCausalLMWithPointer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
    )
    target_dtype = torch.bfloat16 if training_args.bf16 else None
    if target_dtype is not None:
        model.to(dtype=target_dtype, device=training_args.device)
    else:
        model.to(device=training_args.device)
    model.config.use_cache = False
    model.reset_loss_weights(pointer_loss_weight=training_args.pointer_loss_weight, lm_loss_weight=training_args.lm_loss_weight)
    # Build the pointer head after weights are loaded to avoid HF init on meta/ZeRO tensors
    model.ensure_pointer_head_initialized()

    smart_add_special_tokens(tokenizer, model)

    # Initialize vision modules (FastViTHD / MobileCLIP) and wire projector
    model.get_model().initialize_vision_modules(model_args, fsdp=training_args.fsdp)
    vt = model.get_vision_tower()
    vt.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    # AnyRes wiring: make sure model config and arch use consistent settings
    # - image_aspect_ratio controls preprocessing/merging path
    # - image_grid_pinpoints provides candidate sizes for AnyRes
    # - mm_patch_merge_type should be 'spatial' family for AnyRes
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_grid_pinpoints is not None:
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    # Enforce 'spatial' when using AnyRes to match pointer path implementation
    if data_args.image_aspect_ratio == "anyres":
        model.config.mm_patch_merge_type = "spatial"

    # Optionally freeze base and train only pointer head (warmup stage by default)
    if training_args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.multi_patch_pointer_head.parameters():
            p.requires_grad = True

        # Match GUI-Actor warmup: allow learning of pointer token embeddings only.
        # If LM loss is disabled, unfreeze input embeddings and mask grads to pointer tokens.
        try:
            if training_args.lm_loss_weight <= 0.0:
                embed_weight = model.get_input_embeddings().weight
                embed_weight.requires_grad = True
                ptr_ids = [
                    tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN),
                    tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN),
                    tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN),
                ]

                def _mask_pointer_token_grad(grad):
                    mask = torch.zeros_like(grad)
                    for tid in ptr_ids:
                        if 0 <= tid < mask.shape[0]:
                            mask[tid] = 1.0
                    return grad * mask

                embed_weight.register_hook(_mask_pointer_token_grad)
        except Exception
            # If any tokenizer/id edge case occurs, skip masking rather than fail training
            pass

    # Build data module
    data_module = make_data_module(tokenizer=tokenizer, vision_tower=vt, data_args=data_args)

    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Save model
    model.config.use_cache = True
    if trainer.args.should_save:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
