"""
Differential vision encoder used by FastCUA.

This module implements a simple cache for vision tokens produced by the base VLM's
vision tower. Screenshots often change only in a small region between frames; by
detecting which patches changed we can decide to reuse cached tokens or fall back
to a full re-encode when necessary.

The current implementation favours robustness over maximal performance: partial
patch re-encoding is attempted only if the vision tower exposes a dedicated
`encode_patch` method. Otherwise we fall back to re-encoding the full image when
too many patches change. This keeps the integration safe for standard CLIP-based
LLaVA vision towers.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import inspect

import numpy as np
import torch
from PIL import Image

from .patch_utils import compute_patch_diff, extract_patch, extract_patch_window

logger = logging.getLogger(__name__)


class DifferentialVisionEncoder:
    """Cache vision features across frames with simple patch-diff invalidation."""

    def __init__(
        self,
        base_model_or_vision: Any,
        *,
        patch_size: Optional[int] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        diff_threshold: float = 0.05,
        max_changed_patches: int = 10,
        context_radius: int = 1,
        skip_small_updates: bool = False,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        if "change_threshold" in kwargs and kwargs["change_threshold"] is not None:
            diff_threshold = kwargs.pop("change_threshold")
        if kwargs:
            logger.debug("Unused kwargs in DifferentialVisionEncoder: %s", sorted(kwargs.keys()))

        base_model = None
        vision_tower = None
        encode_fn = None

        if hasattr(base_model_or_vision, "encode_images"):
            base_model = base_model_or_vision
            encode_fn = base_model.encode_images
            model = base_model.get_model() if hasattr(base_model, "get_model") else base_model
            if hasattr(model, "get_vision_tower"):
                vision_tower = model.get_vision_tower()
        if vision_tower is None and hasattr(base_model_or_vision, "get_vision_tower"):
            vision_tower = base_model_or_vision.get_vision_tower()
        if vision_tower is None and hasattr(base_model_or_vision, "vision_tower"):
            vision_tower = base_model_or_vision.vision_tower
        if vision_tower is None:
            vision_tower = base_model_or_vision

        if encode_fn is None:
            if hasattr(base_model_or_vision, "__call__"):
                encode_fn = base_model_or_vision
            elif hasattr(vision_tower, "__call__"):
                encode_fn = vision_tower
            else:
                raise ValueError("Could not determine how to encode images for the differential encoder.")

        self.base_model = base_model
        self.vision_tower = vision_tower
        self._encode_full_fn = encode_fn
        self.device = device

        # Infer patch/grid sizes from the vision tower config when possible.
        cfg = getattr(self.vision_tower, "config", None)
        if patch_size is None:
            # Try object-style config
            patch_size = getattr(cfg, "patch_size", None) if cfg is not None else None
            # Try dict-style config
            if patch_size is None and isinstance(cfg, dict):
                patch_size = (
                    cfg.get("image_cfg", {}).get("patch_size")
                    if "image_cfg" in cfg else cfg.get("patch_size")
                )
        if grid_size is None:
            grid_size = getattr(self.vision_tower, "num_patches_per_side", None)
        if patch_size is None and grid_size is not None:
            # Derive patch size from image_size and grid_size
            image_size = getattr(cfg, "image_size", None) if cfg is not None else None
            if image_size is None and isinstance(cfg, dict):
                image_size = (
                    cfg.get("image_cfg", {}).get("image_size")
                    if "image_cfg" in cfg else cfg.get("image_size")
                )
            if image_size is not None and grid_size:
                patch_size = max(image_size // grid_size, 1)
        if grid_size is None and patch_size is not None:
            # Derive grid size from image_size and patch_size
            image_size = getattr(cfg, "image_size", None) if cfg is not None else None
            if image_size is None and isinstance(cfg, dict):
                image_size = (
                    cfg.get("image_cfg", {}).get("image_size")
                    if "image_cfg" in cfg else cfg.get("image_size")
                )
            if image_size is not None and patch_size:
                grid_size = max(image_size // patch_size, 1)

        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_pixels = int(patch_size) if patch_size is not None else None
        if isinstance(grid_size, tuple):
            self.grid_size = tuple(int(x) for x in grid_size)
        elif grid_size is not None:
            self.grid_size = int(grid_size)
        else:
            self.grid_size = None
        self.diff_threshold = float(diff_threshold)
        self.max_changed_patches = int(max_changed_patches)
        self.skip_small_updates = bool(skip_small_updates)
        self.context_radius = int(context_radius)

        self.token_cache: Optional[torch.Tensor] = None
        self.last_image: Optional[np.ndarray] = None
        self.partial_supported = hasattr(self.vision_tower, "encode_patch")
        self._stats: Dict[str, float] = {
            "full_encodes": 0,
            "partial_encodes": 0,
            "cache_hits": 0,
            "skipped_small": 0,
            "total_frames": 0,
            "changed_patches_total": 0,
            "total_patches_total": 0,
        }

    # ---------------------------------------------------------------------
    # Cache utilities
    # ---------------------------------------------------------------------

    def reset_cache(self) -> None:
        self.token_cache = None
        self.last_image = None
        self._stats.update(
            full_encodes=0,
            partial_encodes=0,
            cache_hits=0,
            skipped_small=0,
            total_frames=0,
            changed_patches_total=0,
            total_patches_total=0,
        )

    def reset(self) -> None:  # Backwards alias
        self.reset_cache()

    def stats(self) -> Dict[str, int]:
        return self.get_stats()

    def get_stats(self) -> Dict[str, int]:
        total_patches = self._stats["total_patches_total"]
        avg_change = float(self._stats["changed_patches_total"] / total_patches) if total_patches > 0 else 0.0
        differential = self._stats["partial_encodes"] + self._stats["cache_hits"] + self._stats["skipped_small"]
        return {
            "full_encodes": int(self._stats["full_encodes"]),
            "partial_encodes": int(self._stats["partial_encodes"]),
            "cache_hits": int(self._stats["cache_hits"]),
            "skipped_small": int(self._stats["skipped_small"]),
            "differential_encodes": int(differential),
            "total_frames": int(self._stats["total_frames"]),
            "changed_patches_total": int(self._stats["changed_patches_total"]),
            "total_patches_total": int(total_patches),
            "avg_change_ratio": avg_change,
        }

    # ---------------------------------------------------------------------
    # Encoding entry point
    # ---------------------------------------------------------------------

    def __call__(self, images: Any, return_stats: bool = False):
        return self.encode(images, return_stats=return_stats)

    def encode(
        self,
        images: Any,
        *,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        return_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        tensor = self._prepare_images(images)
        if tensor.ndim != 4:
            raise ValueError(f"Differential encoder expects 4D tensor [B,C,H,W], got {tuple(tensor.shape)}")

        batch = tensor.shape[0]
        image_tensor = tensor[0]
        image_cpu = image_tensor.detach().to(dtype=torch.float32).cpu()
        image_np = image_cpu.permute(1, 2, 0).numpy()

        info: Dict[str, Any] = {"encoding_type": "full", "changed_patches": 0, "total_patches": 0}

        if batch != 1:
            logger.debug("Differential encoder fallback: batch size %d unsupported", batch)
            features = self._full_encode(tensor)
            if return_stats:
                return features, info
            return features

        self._stats["total_frames"] += 1
        grid_h, grid_w = self._grid_shape(image_np.shape[0], image_np.shape[1])
        total_patches = grid_h * grid_w
        info["total_patches"] = total_patches

        if self.token_cache is None or self.last_image is None:
            features = self._full_encode(tensor)
            self._stats["full_encodes"] += 1
            self._stats["total_patches_total"] += total_patches
            self._update_cache(features, image_np)
            if return_stats:
                return features, info
            return features

        if image_np.shape != self.last_image.shape:
            logger.debug("Differential encoder fallback: resolution change %s -> %s", self.last_image.shape, image_np.shape)
            features = self._full_encode(tensor)
            self._stats["full_encodes"] += 1
            self._stats["total_patches_total"] += total_patches
            self._update_cache(features, image_np)
            if return_stats:
                return features, info
            return features

        changed_mask = self._compute_patch_mask(self.last_image, image_np)
        changed_count = int(changed_mask.sum())
        total_patches = changed_mask.size
        info.update({"changed_patches": changed_count, "total_patches": total_patches})
        self._stats["changed_patches_total"] += changed_count
        self._stats["total_patches_total"] += total_patches

        if changed_count == 0:
            self._stats["cache_hits"] += 1
            info["encoding_type"] = "cache"
            self.last_image = image_np
            result = self.token_cache
            if return_stats:
                return result, info
            return result

        if changed_count <= self.max_changed_patches:
            if self.partial_supported:
                features = self._partial_update(tensor, changed_mask)
                if features is not None:
                    self._stats["partial_encodes"] += 1
                    info["encoding_type"] = "partial"
                    self._update_cache(features, image_np)
                    if return_stats:
                        return features, info
                    return features
            if self.skip_small_updates:
                self._stats["skipped_small"] += 1
                info["encoding_type"] = "skip"
                self.last_image = image_np
                result = self.token_cache
                if return_stats:
                    return result, info
                return result

        features = self._full_encode(tensor)
        self._stats["full_encodes"] += 1
        info["encoding_type"] = "full"
        self._update_cache(features, image_np)
        if return_stats:
            return features, info
        return features

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _prepare_images(self, images: Any) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            tensor = images
        elif isinstance(images, Image.Image):
            array = np.array(images)
            tensor = torch.from_numpy(array)
        elif isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images)
        else:
            raise TypeError(f"Unsupported image type for differential encoding: {type(images)!r}")

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected 3D or 4D input, got shape {tuple(tensor.shape)}")

        if tensor.shape[1] not in {1, 3} and tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(0, 3, 1, 2)

        if not tensor.dtype.is_floating_point:
            tensor = tensor.to(dtype=torch.float32)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        if not isinstance(images, torch.Tensor) and self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def _grid_shape(self, height: int, width: int) -> Tuple[int, int]:
        if isinstance(self.grid_size, tuple):
            return self.grid_size
        if self.grid_size is not None:
            return int(self.grid_size), int(self.grid_size)
        if self.patch_pixels is not None and self.patch_pixels > 0:
            size = self.patch_pixels
            return math.ceil(height / size), math.ceil(width / size)
        vt_grid = getattr(self.vision_tower, "num_patches_per_side", None)
        if vt_grid:
            return int(vt_grid), int(vt_grid)
        return height, width  # Fallback: treat individual pixels as patches

    def _full_encode(self, images: torch.Tensor) -> torch.Tensor:
        # Detect model dtype and convert if needed
        if self.base_model is not None and hasattr(self.base_model, "model"):
            try:
                # Try to get dtype from model parameters
                model_dtype = next(self.base_model.model.parameters()).dtype
                if images.dtype != model_dtype:
                    images = images.to(dtype=model_dtype)
            except StopIteration:
                # If no parameters, check for float16 cuda tensors
                if images.device.type == "cuda" and images.dtype != torch.float16:
                    images = images.to(dtype=torch.float16)
        elif images.device.type == "cuda" and images.dtype != torch.float16:
            # Default to float16 for CUDA
            images = images.to(dtype=torch.float16)

        return self._encode_full_fn(images)

    def _update_cache(self, features: torch.Tensor, image_np: np.ndarray) -> None:
        self.token_cache = features.detach()
        self.last_image = image_np

    def _compute_patch_mask(self, prev: np.ndarray, new: np.ndarray) -> np.ndarray:
        if self.patch_pixels is not None:
            patch = self.patch_pixels
        elif self.grid_size is not None and prev.shape[0] == prev.shape[1]:
            patch = max(prev.shape[0] // self.grid_size, 1)
        else:
            # Fallback: treat each pixel as a patch.
            patch = 1
        return compute_patch_diff(prev, new, patch_size=patch, threshold=self.diff_threshold)

    def _partial_update(self, images: torch.Tensor, changed_mask: np.ndarray) -> Optional[torch.Tensor]:
        """Attempt to update only the changed patches if the tower exposes helpers.

        Uses the NEW frame's pixels to recompute tokens for the changed patches.

        Optimizations:
        - Encode all changed patches in a single batched forward pass to minimize
          framework/kernel launch overhead.
        - Apply the mm_projector (if present) in a single batched call.
        """
        vision_tower = self.vision_tower
        encode_patch = getattr(vision_tower, "encode_patch", None)
        if encode_patch is None or self.token_cache is None:
            return None

        try:
            cached = self.token_cache.clone()
        except Exception:
            cached = self.token_cache

        # Determine patch size for extraction
        patch_size = self.patch_pixels
        if patch_size is None:
            cfg = getattr(vision_tower, "config", None)
            patch_size = getattr(cfg, "patch_size", None) if cfg is not None else None
            if patch_size is None and isinstance(cfg, dict):
                patch_size = (
                    cfg.get("image_cfg", {}).get("patch_size")
                    if "image_cfg" in cfg else cfg.get("patch_size")
                )
        if patch_size is None:
            return None

        # Convert the new frame to numpy [H,W,C] in [0,1]
        img_t = images[0].detach().to(dtype=torch.float32).cpu()
        img_np = img_t.permute(1, 2, 0).numpy()
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        # Collect changed patches and their (i, j) indices
        idx_i, idx_j = np.where(changed_mask)
        if idx_i.size == 0:
            return cached

        patches = []
        positions: list[tuple[int, int]] = []
        for i, j in zip(idx_i.tolist(), idx_j.tolist()):
            if self.context_radius > 0:
                patch_np = extract_patch_window(img_np, i, j, patch_size=patch_size, radius=self.context_radius)
            else:
                patch_np = extract_patch(img_np, i, j, patch_size=patch_size)
            patches.append(torch.from_numpy(patch_np).permute(2, 0, 1))  # [C,*,*]
            positions.append((i, j))

        patch_batch = torch.stack(patches, dim=0)  # [B,C,P,P]
        if patch_batch.max() > 1.0:
            patch_batch = patch_batch / 255.0

        # Move once to the tower device/dtype inside encode_patch
        # Try to pass (i,j) positions if the tower supports it
        try:
            sig = inspect.signature(encode_patch)
            if "positions" in sig.parameters:
                pos_tensor = torch.tensor(positions, dtype=torch.long, device=patch_batch.device)
                new_tokens = encode_patch(patch_batch, positions=pos_tensor)  # [B, D]
            else:
                new_tokens = encode_patch(patch_batch)  # [B, D]
        except (ValueError, TypeError):
            new_tokens = encode_patch(patch_batch)  # [B, D]

        # Project to final feature space if needed (batched)
        if self.base_model is not None:
            get_model = getattr(self.base_model, "get_model", None)
            if callable(get_model):
                model_obj = get_model()
                projector = getattr(model_obj, "mm_projector", None)
                if projector is not None:
                    # Ensure dtype/device match projector parameters
                    try:
                        proj_param = next(projector.parameters())
                        new_tokens = new_tokens.to(device=proj_param.device, dtype=proj_param.dtype)
                    except StopIteration:
                        pass
                    if new_tokens.ndim == 2:
                        new_tokens = new_tokens.unsqueeze(1)  # [B,1,D]
                    new_tokens = projector(new_tokens)
                    if new_tokens.ndim == 3 and new_tokens.shape[1] == 1:
                        new_tokens = new_tokens.squeeze(1)  # [B,D]

        # Align to cache dtype/device for scattering
        if cached is not None:
            new_tokens = new_tokens.to(device=cached.device, dtype=cached.dtype)

        # Scatter back into the flat cache
        flat_cache = cached.view(-1, cached.shape[-1])
        grid_h, grid_w = changed_mask.shape
        # Vectorized scatter into flat cache
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=new_tokens.device)
        idxs = pos_tensor[:, 0] * grid_w + pos_tensor[:, 1]
        flat_cache[idxs] = new_tokens

        updated = flat_cache.view_as(cached)
        return updated


__all__ = ["DifferentialVisionEncoder"]
