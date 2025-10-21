"""
Extended FastVLM prediction script with differential vision encoding support.
"""

import argparse
import math
import os
import re

import torch
import numpy as np
from PIL import Image

from differential_vision import DifferentialVisionEncoder
from benchmark import (
    Benchmark,
    MultiSequenceBenchmark,
    add_benchmark_args,
    add_guibench_args,
    GuiBench,
)
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def _natural_sort_key(path: str):
    basename = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", basename)]


def _gather_images_from_dir(directory: str):
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    paths = []
    for entry in os.listdir(directory):
        full = os.path.join(directory, entry)
        if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in supported:
            paths.append(full)
    return sorted(paths, key=_natural_sort_key)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _infer_patch_grid(model, num_tokens: int):
    """
    Infer a (height, width) patch grid from the number of image tokens.

    Preference order:
    1) If the vision tower exposes num_patches_per_side and it matches N=N_side^2, use (N_side, N_side)
    2) Otherwise, factor num_tokens and pick a plausible rectangular grid (h, w) with h*w=num_tokens
       by scanning from sqrt(N) downward.

    Note:
    - This is a heuristic when the exact HxW is unknown (e.g., anyres). Pad mode usually yields square grids.
    - If multiple factor pairs exist, this picks one deterministically; the orientation might be ambiguous.
    """
    tower = model.get_model().get_vision_tower()
    if isinstance(tower, list):
        tower = tower[0]
    side = getattr(tower, "num_patches_per_side", None)
    if isinstance(side, int) and side * side == num_tokens:
        return side, side
    root = int(math.sqrt(num_tokens))
    for height in range(root, 0, -1):
        if num_tokens % height == 0:
            return height, num_tokens // height
    # Fallback: degenerate 1 x N grid (Asly happen)
    return num_tokens, 1


def _extract_image_token_counts(model, image_tensor: torch.Tensor, image_size):
    """
    Run the vision encoder to determine the number of vision tokens per image.

    Returns:
        List[int]: Number of image tokens for each image in the batch.
                   Common shapes:
                     - Tensor [B, N, D] -> returns [N, N, ...] (len B)
                     - Tensor [N, D]    -> returns [N] (single image)
                     - List/Tuple of tensors -> returns [feat.shape[0] for feat in features]
    """
    with torch.inference_mode():
        features = model.encode_images(image_tensor)

    # Normalize different return types to obtain token counts
    if isinstance(features, tuple):
        features = features[0]
    if isinstance(features, torch.Tensor):
        if features.ndim == 3:  # [B, N, D]
            return [features[i].shape[0] for i in range(features.shape[0])]
        if features.ndim == 2:  # [N, D]
            return [features.shape[0]]
    elif isinstance(features, (list, tuple)):
        return [feat.shape[0] for feat in features]

    raise ValueError("Unsupported vision feature shape for heatmap extraction.")


def _build_image_token_mask(sequence_tokens, image_token_counts, total_length, device):
    """
    Construct a boolean mask over the expanded sequence positions that correspond to vision tokens.

    Concept:
    - The textual prompt contains a special IMAGE_TOKEN_INDEX placeholder for each image.
    - During multimodal preprocessing, each IMAGE_TOKEN_INDEX is expanded into many vision tokens (N_i).
    - We simulate this expansion by scanning the (non-expanded) prompt tokens and advancing a cursor:
        * If token == IMAGE_TOKEN_INDEX: mark the next N_i positions as image tokens
        * Else: advance cursor by 1 (textual token occupies one position)

    Args:
        sequence_tokens: list[int]  The unexpanded token ids (contains IMAGE_TOKEN_INDEX placeholders)
        image_token_counts: list[int] Number of vision tokens produced for each image in batch
        total_length: int  The expanded sequence length (e.g., from attention matrices)
        device: torch.device

    Returns:
        torch.BoolTensor of shape [total_length], or None if no image tokens are present.
    """
    if not image_token_counts:
        return None

    mask = torch.zeros(total_length, dtype=torch.bool, device=device)
    cursor = 0
    image_idx = 0

    for token in sequence_tokens:
        if cursor >= total_length:
            break
        if token == IMAGE_TOKEN_INDEX:
            # Expand this placeholder to image_token_counts[image_idx] positions
            if image_idx >= len(image_token_counts):
                break
            count = image_token_counts[image_idx]
            end = min(cursor + count, total_length)
            mask[cursor:end] = True
            cursor = end
            image_idx += 1
        else:
            # Regular text token consumes one position
            cursor += 1

    return mask if mask.any() else None


def _extract_attention_heatmap(model, prompt_ids, image_tensor, image_size, layer_index: int = -1):
    """
    Build an attention heatmap over image patches from a single forward pass.

    What is visualized?
        The attention from the LAST token in the (expanded) prompt sequence to all positions.
        After multimodal expansion, this "last token" is typically the final instruction/punctuation
        token before generation starts. We then restrict that attention vector to the image-token
        positions and reshape to a 2D patch grid.

    Args:
        model:          The multimodal model (LLaVA-style).
        prompt_ids:     Tensor[1, T] of prompt token ids containing IMAGE_TOKEN_INDEX placeholders.
        image_tensor:   Tensor[B=1, ...] preprocessed image tensor for the vision tower.
        image_size:     (width, height) original PIL image size.
        layer_index:    Which transformer layer's attention to use (default: last, i.e. -1).

    Returns:
        torch.FloatTensor [H, W] heatmap of attention over image patches, or None on failure.
    """
    device = _model_device(model)

    # Some attention backends (e.g., Flash-Attn) don't return attention maps.
    # Temporarily force an "eager" attention implementation to get full attentions.
    original_attn_impl = None
    attn_setter = getattr(model, "set_attn_implementation", None)
    used_setter = False
    if attn_setter is not None:
        try:
            current_impl = getattr(model.config, "_attn_implementation", None)
            if current_impl != "eager":
                original_attn_impl = current_impl
                attn_setter("eager")
                used_setter = True
        except Exception:
            pass
    elif hasattr(model.config, "_attn_implementation"):
        current_impl = getattr(model.config, "_attn_implementation")
        if current_impl != "eager":
            original_attn_impl = current_impl
            model.config._attn_implementation = "eager"

    try:
        with torch.inference_mode():
            # Build the multimodal inputs (expanded image tokens, embeddings, masks, etc.)
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = model.prepare_inputs_labels_for_multimodal(
                prompt_ids,         # textual prompt ids with IMAGE_TOKEN_INDEX placeholders
                None,               # position_ids
                None,               # attention_mask
                None,               # past_key_values
                None,               # labels
                image_tensor,       # images
                image_sizes=[image_size],
            )

        # Move to the correct device
        inputs_embeds = inputs_embeds.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            valid_len = int(attention_mask[0].sum().item())
        else:
            valid_len = inputs_embeds.shape[1]

        # Forward with attentions returned
        outputs = model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        # Restore original attention implementation if we changed it
        if original_attn_impl is not None:
            try:
                if used_setter and attn_setter is not None:
                    attn_setter(original_attn_impl)
                elif hasattr(model.config, "_attn_implementation"):
                    model.config._attn_implementation = original_attn_impl
            except Exception:
                pass

    attentions = outputs.attentions
    if not attentions:
        return None

    # Pick the requested layer (supports negative indices)
    layer_count = len(attentions)
    layer_idx = layer_index if layer_index >= 0 else layer_count + layer_index
    layer_idx = max(0, min(layer_idx, layer_count - 1))

    # Attentions: [num_heads, seq_len, seq_len]
    layer_attn = attentions[layer_idx][0]
    layer_attn = layer_attn[:, :valid_len, :valid_len]

    # Sanitize backend anomalies
    layer_attn = torch.nan_to_num(layer_attn, nan=0.0, posinf=0.0, neginf=0.0)

    # Query position: last token in the expanded sequence
    token_pos = layer_attn.shape[1] - 1
    # Collect attention from that token to all positions, average over heads -> [seq_len]
    attn_vec = layer_attn[:, token_pos, :]
    attn_mean = attn_vec.mean(dim=0)
    attn_mean = torch.nan_to_num(attn_mean, nan=0.0, posinf=0.0, neginf=0.0)

    # Map from unexpanded prompt tokens to expanded image-token positions
    sequence_tokens = prompt_ids[0].tolist()
    image_token_counts = _extract_image_token_counts(model, image_tensor, image_size)
    mask = _build_image_token_mask(sequence_tokens, image_token_counts, layer_attn.shape[-1], attn_mean.device)
    if mask is None:
        return None

    # Extract attention over image-token region only
    image_attn = attn_mean[mask]

    # Reshape flat image-token vector to (H, W) patch grid
    grid_h, grid_w = _infer_patch_grid(model, image_attn.shape[0])
    if grid_h * grid_w != image_attn.shape[0]:
        # Defensive check; typically unreachable if factoring succeeded
        return None

    return image_attn.reshape(grid_h, grid_w).to(torch.float32)


def _render_heatmap_overlay(image: Image.Image, heatmap: torch.Tensor, alpha: float, mode: str = "smooth"):
    if heatmap is None:
        return None
    alpha = max(0.0, min(float(alpha), 1.0))
    heat = heatmap
    # Robust normalization with NaN/Inf handling
    finite_mask = torch.isfinite(heat)
    if finite_mask.any():
        minv = heat[finite_mask].min()
        heat = heat - minv
        maxv = heat[finite_mask].max()
        if maxv > 0:
            heat = heat / maxv
    else:
        heat = torch.zeros_like(heat)
    heat = heat.unsqueeze(0).unsqueeze(0)
    interp_mode = "bilinear" if mode != "tile" else "nearest"
    alignable = {"linear", "bilinear", "bicubic", "trilinear"}
    interp_kwargs = {}
    if interp_mode in alignable:
        interp_kwargs["align_corners"] = False
    heat = torch.nn.functional.interpolate(
        heat,
        size=(image.height, image.width),
        mode=interp_mode,
        **interp_kwargs,
    ).squeeze(0).squeeze(0).detach().cpu().numpy()
    # Replace any remaining NaNs/Infs post-interp
    heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)

    base = np.array(image.convert("RGB")).astype(np.float32)
    heat = np.clip(heat, 0.0, 1.0)[..., None]
    alpha_map = heat * alpha
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    overlay = base * (1.0 - alpha_map) + color * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def _resolve_heatmap_path(target, image_path, expects_multiple: bool):
    if target is None:
        return None
    is_directory = (
        expects_multiple
        or target.endswith(os.sep)
        or os.path.isdir(target)
        or os.path.splitext(target)[1] == ""
    )
    if is_directory:
        os.makedirs(target, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(target, f"{stem}_heatmap.png")
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return target


def predict(args):
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    gen_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_path):
        generation_config = os.path.join(model_path, ".generation_config.json")
        os.rename(gen_path, generation_config)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, device=device
    )

    # Optionally force pad mode to avoid anyres batching (>1), which prevents
    # partial differential updates. Pad mode keeps single-image input per frame.
    if getattr(args, "force_pad_aspect", False):
        try:
            setattr(model.config, "image_aspect_ratio", "pad")
        except Exception:
            pass

    diff_encoder = None
    original_encode = getattr(model, "encode_images", None)
    torch_device = torch.device(device)

    # Differential vision is DISABLED by default. Enable only if explicitly requested.
    use_differential = bool(getattr(args, "enable_differential", False)) and not bool(getattr(args, "disable_differential", False))

    if use_differential:
        if original_encode is None:
            raise RuntimeError("Loaded model does not expose encode_images required for differential mode.")

        diff_encoder = DifferentialVisionEncoder(
            model,
            patch_size=args.diff_patch_size,
            diff_threshold=args.diff_threshold,
            max_changed_patches=args.diff_max_changed_patches,
            skip_small_updates=args.diff_skip_small,
            device=torch_device,
        )

        def _call_original(images, image_sizes=None, return_stats=False):
            try:
                return original_encode(images, image_sizes=image_sizes, return_stats=return_stats)
            except TypeError:
                return original_encode(images)

        def encode_images_with_cache(images, image_sizes=None, return_stats=False):
            try:
                return diff_encoder.encode(images, image_sizes=image_sizes, return_stats=return_stats)
            except Exception:
                return _call_original(images, image_sizes=image_sizes, return_stats=return_stats)

        model.encode_images = encode_images_with_cache

    # Build user text according to prompt style
    user_text = args.prompt
    if getattr(args, "prompt_style", "plain") == "pyautogui":
        instr = (
            "Look at the image and predict the exact screen coordinates for this action.\n"
            "Output ONLY valid pyautogui commands. Use normalized coordinates (0.0 to 1.0) based on what you see.\n"
            "Command formats:\n"
            "pyautogui.click(x=<screen_x>, y=<screen_y>)\n"
            "pyautogui.rightClick(x=<screen_x>, y=<screen_y>)\n"
            "pyautogui.doubleClick(x=<screen_x>, y=<screen_y>)\n"
            "pyautogui.moveTo(x=<screen_x>, y=<screen_y>)\n"
            "pyautogui.dragTo(x=<screen_x>, y=<screen_y>)\n"
            "pyautogui.write(message='<text>')\n"
            "pyautogui.press('<key_name>')\n"
            "pyautogui.hotkey('<key1>', '<key2>')\n"
            "pyautogui.scroll(<amount>)\n"
            "pyautogui.hscroll(<amount>)\n\n"
            "Action:"
        )
        user_text = f"{instr}\n{args.prompt}"
    elif getattr(args, "prompt_style", "plain") == "attention_click":
        instr = (
            "Look at the image and predict the exact screen coordinates for this action.\n"
            "Output ONLY valid pyautogui commands. Use normalized coordinates (0.0 to 1.0) based on what you see.\n"
            "Command formats:\n"
            "pyautogui.click x=<screen_x>, y=<screen_y>\n"
            "pyautogui.rightClick x=<screen_x>, y=<screen_y>\n"
            "pyautogui.doubleClick x=<screen_x>, y=<screen_y>\n"
            "pyautogui.moveTo x=<screen_x>, y=<screen_y>\n"
            "pyautogui.dragTo x=<screen_x>, y=<screen_y>\n"
            "pyautogui.scroll <amount>\n"
            "pyautogui.hscroll <amount>\n"
            "pyautogui.write(message='<text>')\n"
            "pyautogui.press('<key_name>')\n"
            "pyautogui.hotkey('<key1>', '<key2>')\n\n"
            "Action:"
        )
        user_text = f"{instr}\n{args.prompt}"

    qs = user_text
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(torch_device)

    # If GuiBench is requested, run it and return early
    if getattr(args, "guibench", False):
        traj_dir = None
        if args.guibench_trajectory:
            traj_dir = os.path.expanduser(args.guibench_trajectory)
            if not (os.path.isdir(traj_dir) and os.path.isfile(os.path.join(traj_dir, "trajectory.json"))):
                raise ValueError(f"Invalid --guibench-trajectory: {traj_dir}")
        elif args.guibench_root:
            picked = GuiBench.pick_trajectory(args.guibench_root, index=int(args.guibench_index or 0))
            if not picked:
                raise ValueError(f"No trajectories found under --guibench-root {args.guibench_root}")
            traj_dir = picked
        else:
            raise ValueError("GuiBench requires --guibench-trajectory or --guibench-root")

        if diff_encoder is not None:
            diff_encoder.reset_cache()

        bench = GuiBench(traj_dir, max_steps=args.guibench_max_steps, verbose=args.guibench_verbose)
        result = bench.run(
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            device=torch_device,
            conv_mode=args.conv_mode,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            context_steps=int(getattr(args, 'guibench_context_steps', 0) or 0),
            custom_prompt=getattr(args, 'guibench_prompt', None),
            attention_click_prompt=bool(getattr(args, 'attention_click', False)),
        )

        print("\n" + "=" * 60)
        print("GUIBENCH SUMMARY")
        print("=" * 60)
        print(f"Trajectory: {result.trajectory_dir}")
        print(f"Steps (total/evaluated): {result.total_steps}/{result.eval_steps}")
        print(f"Expected actions: {result.expected_actions}")
        print(f"Matched actions: {result.matched_actions}")
        print(f"Accuracy: {result.accuracy*100:.1f}%")

        if diff_encoder is not None:
            model.encode_images = original_encode

        if generation_config is not None:
            os.rename(generation_config, os.path.join(model_path, "generation_config.json"))
        return

    sequences = []
    if args.sequence_root:
        root = os.path.expanduser(args.sequence_root)
        if not os.path.isdir(root):
            raise ValueError(f"Sequence root not found: {root}")
        subdirs = sorted(
            [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
            key=_natural_sort_key,
        )
        if args.sequence_limit is not None:
            subdirs = subdirs[: args.sequence_limit]
        for sub in subdirs:
            imgs = _gather_images_from_dir(sub)
            if imgs:
                sequences.append((sub, imgs))
        if not sequences:
            raise ValueError(f"No image sequences found under {root}")
    elif args.image_dir:
        directory = os.path.expanduser(args.image_dir)
        if not os.path.isdir(directory):
            raise ValueError(f"Image directory not found: {directory}")
        images = _gather_images_from_dir(directory)
        if not images:
            raise ValueError(f"No supported image files in {directory}")
        sequences.append((directory, images))
    elif args.image_sequence:
        images = [os.path.expanduser(p) for p in args.image_sequence]
        sequences.append(("image_sequence", images))
    elif args.image_file:
        image_file = os.path.expanduser(args.image_file)
        sequences.append((image_file, [image_file]))
    else:
        raise ValueError("Specify one of --image-file, --image-sequence, --image-dir, or --sequence-root.")

    try:
        first_param = next(model.parameters())
        model_dtype = first_param.dtype
    except StopIteration:
        model_dtype = torch.float16 if torch_device.type != "cpu" else torch.float32

    total_sequences = len(sequences)
    multi_bench = MultiSequenceBenchmark(enabled=bool(args.benchmark))
    for seq_idx, (label, image_paths) in enumerate(sequences, start=1):
        if diff_encoder is not None:
            diff_encoder.reset_cache()
        if total_sequences > 1:
            print(f"\n=== Sequence {seq_idx}/{total_sequences}: {label} ===")
        bench = Benchmark(enabled=args.benchmark)
        bench.reset_sequence()
        for frame_idx, image_path in enumerate(image_paths, start=1):
            frame_start = bench.start_frame()

            image = Image.open(image_path).convert("RGB")
            processed = process_images([image], image_processor, model.config)
            image_tensor = processed[0] if isinstance(processed, (list, tuple)) else processed
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device=torch_device, dtype=model_dtype)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=256,
                    use_cache=True,
                )

            text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            bench.end_frame(frame_start)

            if len(image_paths) > 1:
                print(f"[{frame_idx}/{len(image_paths)}] {os.path.basename(image_path)}: {text}")
            elif total_sequences > 1:
                print(f"{os.path.basename(image_path)}: {text}")
            else:
                print(text)

            if args.save_heatmap:
                expects_multi = total_sequences > 1 or len(image_paths) > 1
                heatmap_path = _resolve_heatmap_path(args.save_heatmap, image_path, expects_multi)
                if heatmap_path:
                    try:
                        heatmap = _extract_attention_heatmap(
                            model,
                            input_ids,  # prompt token ids (with IMAGE_TOKEN_INDEX), visualize attention from the LAST prompt token
                            image_tensor,
                            image.size,
                            layer_index=args.heatmap_layer,
                        )
                        overlay = _render_heatmap_overlay(
                            image, heatmap, alpha=args.heatmap_alpha, mode=getattr(args, "heatmap_mode", "smooth")
                        ) if heatmap is not None else None
                        if overlay is not None:
                            overlay.save(heatmap_path)
                        else:
                            print(f"[heatmap] Skipping {os.path.basename(image_path)} (no vision tokens or attention available).")
                    except Exception as exc:
                        print(f"[heatmap] Failed to generate heatmap for {image_path}: {exc}")

        if diff_encoder is not None:
            stats = diff_encoder.get_stats()
            if args.diff_print_stats:
                print("Differential encoding stats:", stats)
            # Accumulate encoder stats across sequences for an overall summary
            multi_bench.add_encoder_stats(stats)
        bench.print_summary(differential_enabled=bool(diff_encoder))

    if diff_encoder is not None:
        model.encode_images = original_encode

    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, "generation_config.json"))

    # Print overall multi-sequence summary if benchmarking was enabled
    multi_bench.print_overall(differential_enabled=bool(diff_encoder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None, help="Location of a single image.")
    parser.add_argument(
        "--image-sequence",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of image paths forming a sequence.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing a single image sequence.",
    )
    parser.add_argument(
        "--sequence-root",
        type=str,
        default=None,
        help="Directory whose subdirectories each contain image sequences.",
    )
    parser.add_argument(
        "--sequence-limit",
        type=int,
        default=None,
        help="Process at most this many sequences when using --sequence-root.",
    )
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt or action text.")
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["plain", "pyautogui", "attention_click"],
        default="plain",
        help="How to construct the prompt from the action text.",
    )
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    # Differential vision flags
    parser.add_argument(
        "--enable-differential",
        action="store_true",
        help="Enable differential vision encoding (disabled by default).",
    )
    parser.add_argument(
        "--disable-differential",
        action="store_true",
        help="Deprecated: differential is disabled by default; use --enable-differential to turn it on.",
    )
    parser.add_argument("--diff-threshold", type=float, default=0.05, help="Patch change threshold (0-1).")
    parser.add_argument(
        "--diff-max-changed-patches",
        type=int,
        default=50,
        help="Maximum changed patches before falling back to full encode.",
    )
    parser.add_argument("--diff-patch-size", type=int, default=None, help="Override the vision patch size.")
    parser.add_argument(
        "--diff-skip-small",
        action="store_true",
        help="Reuse cached tokens when only a few patches change.",
    )
    parser.add_argument(
        "--diff-print-stats",
        action="store_true",
        help="Print differential encoder statistics after processing.",
    )
    parser.add_argument(
        "--force-pad-aspect",
        action="store_true",
        help="Force model.config.image_aspect_ratio='pad' to enable partial differential updates.",
    )
    parser.add_argument(
        "--save-heatmap",
        type=str,
        default=None,
        help="Save an attention heatmap overlay to this file or directory.",
    )
    parser.add_argument(
        "--heatmap-layer",
        type=int,
        default=-1,
        help="Transformer layer index for attention heatmap (default: last layer).",
    )
    parser.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.45,
        help="Alpha blending for heatmap overlay (0-1).",
    )
    parser.add_argument(
        "--heatmap-mode",
        type=str,
        choices=["smooth", "tile"],
        default="smooth",
        help="Heatmap upsampling mode: 'smooth' (bilinear) or 'tile' (nearest).",
    )
    add_benchmark_args(parser)
    add_guibench_args(parser)
    args = parser.parse_args()

    predict(args)
