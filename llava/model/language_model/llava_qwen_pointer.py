import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List, Optional, Tuple, Union

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_qwen import (
    LlavaQwen2ForCausalLM as _BaseLlavaQwen2ForCausalLM,
    LlavaConfig,
)
from ..llava_arch import LlavaMetaModel
from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_POINTER_PAD_TOKEN,
)
from llava.mm_utils import get_anyres_image_grid_shape


class VisionHead_MultiPatch(nn.Module):
    """Multi-query -> visual-patch pointer head.

    Attends from target token hidden states (queries) to visual token embeddings (keys/values)
    using a simple 2-layer MLP projection and dot-product attention. Supervision supports
    either single-patch (CE) or multi-patch (KL to a row-normalized binary mask) training.
    """

    def __init__(self, d_model: int, projection_dim: int, num_attention_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.projection_enc = nn.Sequential(
            nn.Linear(d_model, projection_dim), nn.GELU(), nn.Linear(projection_dim, d_model)
        )
        self.projection_dec = nn.Sequential(
            nn.Linear(d_model, projection_dim), nn.GELU(), nn.Linear(projection_dim, d_model)
        )
        # Light contextualization of visual embeddings via self-attention.
        self.layer_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_state_enc: torch.Tensor,  # [n_enc, d_model] visual embeddings (input-space)
        hidden_state_dec: torch.Tensor,  # [n_dec, d_model] target token hidden states (output-space)
        labels: Optional[torch.Tensor] = None,  # [n_dec, n_enc] binary mask of patches in bbox
        do_single_patch: bool = False,
    ):
        # Contextualize visual embeddings slightly via self-attention
        enc_input = hidden_state_enc.unsqueeze(0)  # [1, n_enc, d_model]
        attn_output, _ = self.self_attention(
            query=enc_input, key=enc_input, value=enc_input, need_weights=False
        )
        hidden_state_enc_ctx = self.layer_norm(enc_input + self.dropout(attn_output)).squeeze(0)

        proj_enc = self.projection_enc(hidden_state_enc_ctx)  # [n_enc, d_model]
        proj_dec = self.projection_dec(hidden_state_dec)      # [n_dec, d_model]

        scaling = self.d_model ** 0.5
        patch_logits = torch.matmul(proj_dec, proj_enc.transpose(0, 1)) / scaling  # [n_dec, n_enc]
        attn_weights = F.softmax(patch_logits, dim=-1)

        loss = None
        if labels is not None and not do_single_patch:
            epsilon = 1e-8
            labels_float = labels.float()
            target_dist = labels_float / (labels_float.sum(dim=-1, keepdim=True) + epsilon)
            pred_log_probs = F.log_softmax(patch_logits, dim=-1)
            loss = F.kl_div(pred_log_probs, target_dist, reduction="batchmean")
        elif labels is not None and do_single_patch:
            loss = F.cross_entropy(patch_logits, labels)

        return attn_weights, loss


class LlavaQwen2ForCausalLMWithPointer(_BaseLlavaQwen2ForCausalLM):
    """LLaVA-Qwen2 with an additional pointer head for visual grounding.

    This mirrors GUI-Actor's pointer training design but plugs into the LLaVA/FastVLM
    stack (vision_tower + mm_projector). It computes LM loss as usual, and optionally
    adds a pointer supervision term by attending from target tokens to the inserted
    visual token embeddings.
    """

    config_class = LlavaConfig

    def __init__(self, config):
        super().__init__(config)
        hidden = config.hidden_size
        # Lazily build pointer head after pretrained weights are loaded to avoid
        # HF initialize-missing-keys interacting with DS ZeRO/init contexts.
        self._pointer_hidden_size = hidden
        self.multi_patch_pointer_head: Optional[VisionHead_MultiPatch] = None
        self.pointer_loss_weight: float = 1.0
        self.lm_loss_weight: float = 1.0
        # post_init already called by base

    def ensure_pointer_head_initialized(self):
        if self.multi_patch_pointer_head is None:
            ph = VisionHead_MultiPatch(self._pointer_hidden_size, self._pointer_hidden_size)
            # Match dtype/device to the base model
            try:
                ref_param = next(self.parameters())
                ph.to(dtype=ref_param.dtype, device=ref_param.device)
            except StopIteration:
                pass
            self.multi_patch_pointer_head = ph

    def reset_loss_weights(self, pointer_loss_weight: float, lm_loss_weight: float):
        self.pointer_loss_weight = pointer_loss_weight
        self.lm_loss_weight = lm_loss_weight

    def _find_pointer_targets(self, labels_row: torch.Tensor, pointer_pad_id: int) -> torch.Tensor:
        """Return 1D LongTensor of positions where labels equal the pointer pad token id."""
        return torch.nonzero(labels_row == pointer_pad_id, as_tuple=False).squeeze(-1)

    def _compute_visual_grid_size(self) -> Optional[int]:
        vt = self.get_vision_tower()
        if hasattr(vt, "num_patches_per_side"):
            return vt.num_patches_per_side
        return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        # Pointer supervision (optional)
        coordinates: Optional[List[List[Tuple[float, float]]]] = None,  # list per batch: list of (x,y)
        multi_patch_labels: Optional[List[torch.Tensor]] = None,         # list per batch: [n_target, n_visual]
        bboxes: Optional[List[Optional[List[float]]]] = None,            # list per batch: [x1,y1,x2,y2] normalized
        if_multi_patch: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Ensure the pointer head exists (created post-load)
        self.ensure_pointer_head_initialized()

        # Always prepare multimodal embeds here to also collect image_features for pointer loss
        if inputs_embeds is None:
            (
                _input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )
        else:
            new_input_embeds, new_labels = inputs_embeds, labels

        # Run LM forward to get logits and hidden states.
        # Always pass labels so HF constructs an outputs with a .loss field, then gate by lm_loss_weight below.
        hf_labels = new_labels
        outputs = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_input_embeds,
            labels=hf_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need hidden states for pointer head
            return_dict=True,
        )

        # Respect lm_loss_weight: treat LM loss as disabled when weight <= 0
        lm_loss = outputs.loss if (self.lm_loss_weight is None or self.lm_loss_weight > 0) else None
        last_hidden = outputs.hidden_states[-1]  # [bs, seq_len, hidden]

        # Pointer supervision: infer target positions from labels and use the input-time visual embeddings.
        # If pointer supervision is requested (pointer_loss_weight > 0), we require sufficient inputs.
        pointer_loss = None
        if self.pointer_loss_weight > 0:
            if new_labels is None:
                raise ValueError("Pointer supervision requested but labels are None; ensure labels are provided.")
            if not hasattr(self.config, "pointer_pad_token_id") or self.config.pointer_pad_token_id is None:
                raise ValueError("Pointer supervision requested but config.pointer_pad_token_id is not set.")
            bs = new_labels.shape[0]
            pointer_losses: List[torch.Tensor] = []
            real_target_count = 0
            # Reuse the exact image features used during multimodal preparation for alignment
            if not hasattr(self, "_last_visual_embeds") or self._last_visual_embeds is None:
                raise ValueError(
                    "Pointer supervision requested but _last_visual_embeds not set. "
                    "Ensure images are passed and prepare_inputs_labels_for_multimodal is used."
                )
            img_feats_list = self._last_visual_embeds

            ptr_token_id = getattr(self.config, "pointer_pad_token_id", None)
            grid_N = self._compute_visual_grid_size()

            for i in range(bs):
                # Locate pointer token positions from the final labels (after image insertion and any truncation/padding)
                # to ensure indices align with last_hidden's sequence length.
                if new_labels is None or ptr_token_id is None:
                    raise ValueError("Pointer supervision requested but new_labels or pointer_pad_token_id is missing.")
                labels_row = new_labels[i]
                target_positions = torch.nonzero(labels_row == ptr_token_id, as_tuple=False).squeeze(-1)
                if target_positions.numel() == 0:
                    raise ValueError(
                        "Pointer supervision requested but no pointer tokens were found in labels. "
                        "Ensure the dataset injected <|pointer_start|><|pointer_pad|><|pointer_end|> into assistant text."
                    )
                real_target_count += int(target_positions.numel())

                # target hidden states (queries). Guard against any stale/bad indices.
                seq_len = last_hidden.shape[1]
                # Keep only in-bounds indices and move to correct device for indexing
                target_positions = target_positions[target_positions >= 0]
                target_positions = target_positions[target_positions < seq_len]
                if target_positions.numel() == 0:
                    raise ValueError("Pointer supervision: all target positions are out of bounds after truncation.")
                target_hidden = last_hidden[i, target_positions.to(last_hidden.device)]  # [n_target, hidden]

                # Visual embeddings (keys). Prefer re-encoded image features.
                # img_feats_list[i] is [N, hidden] or [M, N, hidden] if list-handled upstream
                visual_embeds = img_feats_list[i]
                if visual_embeds.ndim == 3:
                    b, n, d = visual_embeds.shape
                    visual_embeds = visual_embeds.reshape(b * n, d)
                if visual_embeds.numel() == 0:
                    raise ValueError("Pointer supervision requested but visual embeddings are empty for a sample.")

                # Optional multi-patch labels
                labels_multi = None
                if multi_patch_labels is not None and multi_patch_labels[i] is not None:
                    labels_multi = multi_patch_labels[i].to(visual_embeds.device)
                elif coordinates is not None and coordinates[i] is not None:
                    # Tile-aware mapping for AnyRes: compute the tile grid (num_patch_width/height)
                    # and offset by the base downsampled tokens.
                    # Fallback to single-grid mapping if AnyRes metadata is unavailable.
                    vlen = visual_embeds.shape[0]
                    base_per_side = grid_N  # e.g., 16 for 1024/patch vision towers
                    base_tokens = None if base_per_side is None else (base_per_side * base_per_side)

                    num_tile_w = None
                    num_tile_h = None
                    tile_side = None
                    try:
                        # Mirror logic in llava_arch: derive the tower tile size
                        vt = self.get_vision_tower()
                        if hasattr(vt, 's2_image_size'):
                            tile_side = vt.s2_image_size
                        elif isinstance(vt.config, dict):
                            tile_side = vt.config.get("image_cfg", {}).get("image_size", None)
                        else:
                            tile_side = getattr(vt.config, 'image_size', None)

                        # image_sizes[i] is [W, H] per PointerCollator
                        if image_sizes is not None and tile_side is not None and hasattr(self.config, 'image_grid_pinpoints') and self.config.image_grid_pinpoints is not None:
                            from llava.mm_utils import get_anyres_image_grid_shape
                            iw, ih = image_sizes[i]
                            num_tile_w, num_tile_h = get_anyres_image_grid_shape((iw, ih), self.config.image_grid_pinpoints, tile_side)
                    except Exception:
                        # Fall back to flat mapping below if anything goes wrong
                        num_tile_w, num_tile_h = None, None

                    # Validate normalized coordinates
                    for (x, y) in coordinates[i]:
                        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
                            raise ValueError(f"Coordinates must be normalized to [0,1], got ({x}, {y}).")

                    # Use the actual number of target tokens as rows; coordinates may contain extra noise
                    tgt_cnt = int(target_hidden.shape[0])
                    labels_multi = torch.zeros((tgt_cnt, vlen), device=visual_embeds.device, dtype=torch.float32)
                    coords_list = coordinates[i]
                    if len(coords_list) != tgt_cnt:
                        # Keep it quiet but robust: align to the min length
                        use_n = min(len(coords_list), tgt_cnt)
                        if use_n == 0:
                            raise ValueError("Pointer supervision: zero effective targets after aligning coordinates to tokens.")
                        coords_iter = list(coords_list)[:use_n]
                    else:
                        coords_iter = coords_list

                    if num_tile_w is not None and num_tile_h is not None and base_tokens is not None:
                        # Expect vlen = base_tokens + num_tile_w*num_tile_h*base_tokens in 'spatial' path.
                        # Proceed best-effort even if it differs (e.g., due to different merge type).
                        eps = 1e-6
                        for r, (x, y) in enumerate(coords_iter):
                            xf = float(x)
                            yf = float(y)
                            # Continuous tile coordinates
                            x_rel = xf * num_tile_w
                            y_rel = yf * num_tile_h
                            tile_x = int(min(max(math.floor(x_rel), 0), num_tile_w - 1))
                            tile_y = int(min(max(math.floor(y_rel), 0), num_tile_h - 1))
                            # Local fractional position within tile
                            lx = x_rel - tile_x
                            ly = y_rel - tile_y
                            # Map to patch indices within the tile grid (base_per_side x base_per_side)
                            px = int(min(max(math.floor(lx * base_per_side), 0), base_per_side - 1))
                            py = int(min(max(math.floor(ly * base_per_side), 0), base_per_side - 1))

                            # Compute global index within the concatenated visual embeddings
                            # Layout in llava_arch (spatial): [base 256] + flatten over dims (tile_y, patch_y, tile_x, patch_x)
                            idx_local = (((tile_y * base_per_side + py) * num_tile_w) + tile_x) * base_per_side + px
                            idx_global = base_tokens + idx_local
                            if 0 <= idx_global < vlen:
                                labels_multi[r, idx_global] = 1.0

                            # Boundary handling: if the click lies exactly on a tile boundary, also mark the adjacent tile's edge patch
                            # Left/right boundaries
                            # Right neighbor (close to next tile boundary)
                            if abs((xf * num_tile_w) - (tile_x + 1)) < eps and tile_x + 1 < num_tile_w:
                                # Right tile, leftmost column
                                t2x = tile_x + 1
                                p2x = 0
                                idx_local_2 = (((tile_y * base_per_side + py) * num_tile_w) + t2x) * base_per_side + p2x
                                idx_global_2 = base_tokens + idx_local_2
                                if 0 <= idx_global_2 < vlen:
                                    labels_multi[r, idx_global_2] = 1.0
                            # Left neighbor (close to previous tile boundary)
                            if abs((xf * num_tile_w) - tile_x) < eps and tile_x - 1 >= 0:
                                t2x = tile_x - 1
                                p2x = base_per_side - 1
                                idx_local_2 = (((tile_y * base_per_side + py) * num_tile_w) + t2x) * base_per_side + p2x
                                idx_global_2 = base_tokens + idx_local_2
                                if 0 <= idx_global_2 < vlen:
                                    labels_multi[r, idx_global_2] = 1.0
                            # Top/bottom boundaries
                            if abs((yf * num_tile_h) - (tile_y + 1)) < eps and tile_y + 1 < num_tile_h:
                                t2y = tile_y + 1
                                p2y = 0
                                idx_local_2 = (((t2y * base_per_side + p2y) * num_tile_w) + tile_x) * base_per_side + px
                                idx_global_2 = base_tokens + idx_local_2
                                if 0 <= idx_global_2 < vlen:
                                    labels_multi[r, idx_global_2] = 1.0
                            if abs((yf * num_tile_h) - tile_y) < eps and tile_y - 1 >= 0:
                                t2y = tile_y - 1
                                p2y = base_per_side - 1
                                idx_local_2 = (((t2y * base_per_side + p2y) * num_tile_w) + tile_x) * base_per_side + px
                                idx_global_2 = base_tokens + idx_local_2
                                if 0 <= idx_global_2 < vlen:
                                    labels_multi[r, idx_global_2] = 1.0
                    else:
                        # Fallback: original single-grid mapping over the entire visual token sequence
                        infer_grid = base_per_side
                        if infer_grid is None:
                            # Best-effort square inference
                            root = int((vlen) ** 0.5)
                            if root * root == vlen:
                                infer_grid = root
                        if infer_grid is None or infer_grid <= 0:
                            raise ValueError(
                                "Coordinates provided but grid size is unknown and cannot be inferred from visual token count."
                            )
                        for r, (x, y) in enumerate(coords_iter):
                            x_idx = min(max(int(float(x) * infer_grid), 0), infer_grid - 1)
                            y_idx = min(max(int(float(y) * infer_grid), 0), infer_grid - 1)
                            idx = y_idx * infer_grid + x_idx
                            if 0 <= idx < vlen:
                                labels_multi[r, idx] = 1.0
                elif bboxes is not None and bboxes[i] is not None and grid_N is not None:
                    # Build binary mask over the grid for a bbox, replicate per target
                    vlen = visual_embeds.shape[0]
                    # if we don't know the number of pointer targets, default to 1 row
                    tgt_cnt = target_hidden.shape[0]
                    labels_multi = torch.zeros((tgt_cnt, vlen), device=visual_embeds.device, dtype=torch.float32)
                    x1, y1, x2, y2 = bboxes[i]
                    x1i = min(max(int(x1 * grid_N), 0), grid_N - 1)
                    y1i = min(max(int(y1 * grid_N), 0), grid_N - 1)
                    x2i = min(max(int(x2 * grid_N), 0), grid_N - 1)
                    y2i = min(max(int(y2 * grid_N), 0), grid_N - 1)
                    for yy in range(min(y1i, y2i), max(y1i, y2i) + 1):
                        for xx in range(min(x1i, x2i), max(x1i, x2i) + 1):
                            idx = yy * grid_N + xx
                            if idx < vlen:
                                labels_multi[:, idx] = 1.0
                else:
                    raise ValueError(
                        "Pointer supervision requested but neither multi_patch_labels, coordinates, nor bboxes are provided for a sample."
                    )

                # Ensure label tensor shape matches [n_target, n_enc]
                if labels_multi is not None:
                    n_dec = int(target_hidden.shape[0])
                    n_enc = int(visual_embeds.shape[0])
                    # Align rows (targets)
                    if labels_multi.shape[0] != n_dec:
                        new_rows = min(n_dec, labels_multi.shape[0])
                        target_hidden = target_hidden[:new_rows]
                        labels_multi = labels_multi[:new_rows, :]
                        n_dec = new_rows
                    # Align cols (visual tokens)
                    if labels_multi.shape[1] != n_enc:
                        if labels_multi.shape[1] < n_enc:
                            pad_cols = n_enc - labels_multi.shape[1]
                            labels_multi = torch.cat(
                                [labels_multi, labels_multi.new_zeros((labels_multi.shape[0], pad_cols))], dim=1
                            )
                        else:
                            labels_multi = labels_multi[:, :n_enc]

                # Compute pointer loss for this sample
                attn_scores, loss_v = self.multi_patch_pointer_head(
                    visual_embeds,
                    target_hidden,
                    labels=labels_multi,
                )
                if loss_v is None:
                    raise ValueError("Pointer loss computation returned None unexpectedly.")
                pointer_losses.append(loss_v)

            if bs > 0 and real_target_count == 0:
                raise ValueError("Pointer supervision requested but no pointer tokens found in the entire batch.")
            if len(pointer_losses) == 0:
                raise ValueError("Pointer supervision requested but no per-sample pointer losses were computed.")
            pointer_loss = torch.stack(pointer_losses).mean()

        # Combine losses
        total_loss = None
        if lm_loss is not None and pointer_loss is not None:
            total_loss = self.lm_loss_weight * lm_loss + self.pointer_loss_weight * pointer_loss
        elif lm_loss is None:
            # LM disabled; require pointer loss to be present
            if pointer_loss is None:
                raise ValueError("Both LM loss disabled and pointer loss missing; cannot proceed without a loss.")
            total_loss = pointer_loss
        else:
            total_loss = lm_loss

        # Always return HF output dict and set combined loss (must be non-None by logic above)
        outputs.loss = total_loss
        return outputs


# Keep registration consistent with base class for AutoModel/API parity
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("llava_qwen2", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaQwen2ForCausalLMWithPointer)
