"""
Qwen3-VL attention heatmap + click targeting example.

This script extracts attention heatmaps and derives click targets from
high-attention regions. Only the attention-based clicking flow is kept.
"""

import os
import re
from datetime import datetime
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from click_targeter import ClickTargeter


def _build_qwen_inputs(processor, image: Image.Image, prompt: str, device: str = "cuda"):
    """Build text + image inputs for Qwen3-VL via the processor.

    Returns a dict with input_ids, attention_mask, pixel_values, image_grid_thw on the target device.
    """
    # Build chat text with an image placeholder using the processor's chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "file://placeholder"},  # URL not used; just triggers image token
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Get the text prompt (no tokenization here)
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and process image together; ensures image_grid_thw is returned
    inputs = processor(
        text=[chat_text], images=[image], return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)

    # Move tensors to device
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    return inputs


def _extract_qwen_heatmap(
    model,
    processor,
    inputs: dict,
    layer_index: int = -1,
    query_strategy: str = "last_text",
) -> Optional[torch.Tensor]:
    """Extract an attention heatmap over visual tokens for Qwen3-VL.

    Returns a tensor of shape [H_grid, W_grid] on the same device as model.
    """
    device = next(model.parameters()).device
    model.set_attn_implementation("eager")
        
    # Forward to collect attentions (no cache to simplify shapes)
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )

    if not getattr(outputs, "attentions", None):
        return None

    attns = outputs.attentions
    lyr = layer_index if layer_index >= 0 else len(attns) + layer_index
    # attns[lyr]: (batch, heads, q_len, kv_len)
    layer = attns[lyr][0]

    # Determine query position
    attn_mask = inputs.get("attention_mask")
    if attn_mask is not None:
        valid_len = int(attn_mask[0].sum().item())
    else:
        valid_len = inputs["input_ids"].shape[1]

    # Build mask for image tokens (processor exposes image_token_id)
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            getattr(processor, "image_token", "<|image_pad|>")
        )
    token_ids = inputs["input_ids"][0]
    img_mask = token_ids == image_token_id

    # Choose query index/indices
    if query_strategy == "cls":
        q_indices = [0]
    elif query_strategy.startswith("mean_last_"):
        try:
            k = int(query_strategy.split("_")[-1])
        except Exception:
            k = 4
        non_img = (~img_mask)[:valid_len]
        idx = torch.nonzero(non_img, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            q_indices = list(range(max(0, valid_len - k), valid_len))
        else:
            k = max(1, min(k, idx.numel()))
            q_indices = [int(x.item()) for x in idx[-k:]]
    elif query_strategy == "last_text":
        non_img = (~img_mask)[:valid_len]
        idx = torch.nonzero(non_img, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            q_indices = [int(idx[-1].item())]
        else:
            q_indices = [max(0, valid_len - 1)]
    else:
        q_indices = [max(0, valid_len - 1)]

    # Average over heads and (optionally) over multiple query positions -> vector over kv_len
    if len(q_indices) == 1:
        attn_vec = layer[:, q_indices[0], :].mean(dim=0)
    else:
        q_idx = torch.tensor(q_indices, device=layer.device, dtype=torch.long)
        attn_vec = layer[:, q_idx, :].mean(dim=0).mean(dim=0)

    if img_mask.sum().item() == 0:
        print("No image tokens found, trying fallback...")

    vision_attn = attn_vec[img_mask]

    # Derive grid size from image_grid_thw and merge_size
    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        # Cannot reshape reliably
        print("Cannot reshape reliably because the image grid is not present")
        return None
    t, h, w = grid_thw[0].tolist()
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    # tokens are merged spatially by merge_size^2
    h_grid = max(1, h // merge_size)
    w_grid = max(1, w // merge_size)
    expected = h_grid * w_grid

    if vision_attn.numel() != expected:
        print("Cannot reshape reliably because the number of tokens does not match the expected number of tokens")
        # As a fallback, try best factorization
        n = vision_attn.numel()
        # Prefer a shape close to square
        import math

        g_h = int(math.sqrt(n))
        while g_h > 1 and n % g_h != 0:
            g_h -= 1
        g_w = n // g_h
        h_grid, w_grid = g_h, g_w

    heatmap = vision_attn.reshape(h_grid, w_grid)
    return heatmap.to(device)

def demo_attention_clicking(
    model_id: str,
    image_path: str,
    prompt: str = "Click on the button",
    device: str = "cuda",
    compare_prediction: bool = False,
    interpolation_mode: str = "bilinear",
    save_raw_grid: bool = True,
    layer_index: int = -1,
):
    """Demonstrate attention-based click targeting with Qwen3-VL.

    Args:
        model_id: HF repo or local path for Qwen3-VL
        image_path: Path to screenshot/UI image
        prompt: Instruction prompt
        device: Compute device
        compare_prediction: Optionally decode model output for comparison
        interpolation_mode: 'bilinear' or 'nearest'
        save_raw_grid: Save the raw attention grid to a txt file
        layer_index: Which transformer layer to read attentions from
    """
    # Load model and processor
    print("Loading Qwen3-VL model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Load image
    print("Processing image...")
    image = Image.open(image_path).convert("RGB")

    # Build inputs for forward + attentions
    inputs = _build_qwen_inputs(processor, image, prompt, device=device)

    # Extract attention heatmap
    print("Extracting attention heatmap...")
    heatmap = _extract_qwen_heatmap(model, processor, inputs, layer_index=layer_index, query_strategy="last_text")
    if heatmap is None:
        print("Failed to extract attention heatmap")
        return None
    print(f"Heatmap shape: {tuple(heatmap.shape)}")

    # Optionally save raw grid values
    if save_raw_grid:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem, _ = os.path.splitext(os.path.basename(image_path))
        grid_out = f"{stem}_attention_grid_{timestamp}.txt"
        with open(grid_out, "w") as f:
            f.write(f"Raw {tuple(heatmap.shape)} attention grid:\n\n")
            # Ensure numpy-compatible dtype
            grid_np = heatmap.to(torch.float32).detach().cpu().numpy()
            for i in range(grid_np.shape[0]):
                for j in range(grid_np.shape[1]):
                    f.write(f"{grid_np[i, j]:.6f} ")
                f.write("\n")
        print(f"Saved raw attention grid to {grid_out}")

    # Generate click candidates and overlay
    print("Generating click candidates...")
    targeter = ClickTargeter(threshold=0.5, min_area_ratio=0.001, max_candidates=5)
    candidates = targeter.generate_candidates(
        heatmap, image.size, adaptive_threshold=True, interpolation_mode=interpolation_mode
    )
    print(f"Found {len(candidates)} click candidates")

    # Save heatmap overlay regardless of candidates
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem, _ = os.path.splitext(os.path.basename(image_path))
        heatmap_out = f"{stem}_heatmap_{timestamp}.png"
        overlay_only = targeter.visualize_candidates(
            image, candidates=[], heatmap=heatmap, alpha=0.35, interpolation_mode=interpolation_mode
        )
        overlay_only.save(heatmap_out)
        print(f"Saved heatmap overlay to {heatmap_out}")
    except Exception as e:
        print(f"Failed to save heatmap overlay: {e}")

    predicted_click = None
    if compare_prediction:
        predicted_click = compare_with_model_prediction(model, processor, inputs)
        if predicted_click:
            print(f"Decoded model click prediction: {predicted_click}")
        else:
            print("Model prediction did not include explicit click coordinates")

    if candidates:
        click_point = targeter.select_click_point(candidates, strategy="center_weighted")
        print(f"Best click point: {click_point}")

        vis_image = targeter.visualize_candidates(
            image, candidates, heatmap, alpha=0.3, interpolation_mode=interpolation_mode
        )
        output_path = image_path.replace(".png", "_attention_clicks.png")
        vis_image.save(output_path)
        print(f"Saved visualization to {output_path}")
        return click_point, candidates, predicted_click

    print("No click candidates found")
    return None, [], predicted_click


def compare_with_model_prediction(model, processor, inputs: dict) -> Optional[Tuple[int, int]]:
    """Generate text and try to parse a click(x,y) from output for comparison."""
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            max_new_tokens=128,
            do_sample=False,
            temperature=0.1,
        )

    # Trim prefix to only the generated continuation
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    prediction = processor.post_process_image_text_to_text(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(f"Model prediction: {prediction}")

    match = re.search(r"click\((\d+),\s*(\d+)\)", prediction)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-VL attention heatmap demo")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Click on the most important button")
    parser.add_argument("--device", type=str, default="cuda")
    # Attention mode options
    parser.add_argument("--compare-model", action="store_true")
    parser.add_argument("--interpolation", type=str, default="bilinear", choices=["bilinear", "nearest"])
    parser.add_argument("--no-save-grid", action="store_true")
    parser.add_argument("--layer", type=int, default=-1)

    args = parser.parse_args()

    result = demo_attention_clicking(
        args.model_id,
        args.image,
        args.prompt,
        args.device,
        args.compare_model,
        args.interpolation,
        not args.no_save_grid,
        args.layer,
    )
    if result is not None:
        click_point, candidates, predicted_click = result
        if click_point is None and not candidates:
            print("\nSummary:\n  No attention-based click candidates were found.")
        else:
            print("\nSummary:")
            print(f"  Selected click point: {click_point}")
            print(f"  Total candidates: {len(candidates)}")
            for i, cand in enumerate(candidates[:3]):
                print(
                    f"  Candidate {i+1}: center={cand.center}, score={cand.attention_score:.3f}"
                )
        if args.compare_model:
            if predicted_click:
                print(f"  Model predicted click: {predicted_click}")
            else:
                print("  Model did not produce a parseable click prediction")
