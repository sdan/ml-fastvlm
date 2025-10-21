"""
Single-pass attention-based clicking with bounding box injection for Qwen3-VL.

This implementation extracts attention during generation, computes bounding boxes,
and injects them into the context for the model to select from - all in one pass.
"""

import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from click_targeter import ClickTargeter


def _build_structured_prompt(user_instruction: str, display_width: int = 1920, display_height: int = 1080) -> str:
    """Build a structured prompt that sets up for box injection.
    
    Args:
        user_instruction: The user's instruction for what to do
        display_width: Display width in pixels (default: 1920)
        display_height: Display height in pixels (default: 1080)
    """
    # Hardcoded structured prompt for GUI actions
    return f"""This is an interface to a desktop GUI. The screen's resolution is {display_width}x{display_height}.

The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.

User instruction: {user_instruction}

Take the next action"""


def _extract_attention_at_token(
    model,
    processor,
    inputs: dict,
    target_token: str = "action",
    layer_index: int = -1,
    use_last_occurrence: bool = True,
) -> Optional[torch.Tensor]:
    """Extract attention heatmap when a specific token is generated.
    
    Args:
        model: The Qwen3-VL model
        processor: The processor
        inputs: Model inputs
        target_token: Token to trigger attention extraction at
        layer_index: Which layer to extract from
        use_last_occurrence: If True, use last occurrence of token; if False, use first
        
    Returns:
        Attention heatmap as tensor or None
    """
    device = next(model.parameters()).device
    
    # Find the position of the target token in the input
    token_ids = inputs["input_ids"][0]
    tokenizer = processor.tokenizer

    # Tokenize the target word to find its ID
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)

    # Find all occurrences of this token in the sequence
    occurrences = []
    for i in range(len(token_ids) - len(target_ids) + 1):
        if all(token_ids[i + j] == target_ids[j] for j in range(len(target_ids))):
            occurrences.append(i + len(target_ids) - 1)  # Position of last token in target

    if not occurrences:
        print(f"Target token '{target_token}' not found in sequence")
        return None
    
    # Use last occurrence if requested, otherwise first
    target_pos = occurrences[-1] if use_last_occurrence else occurrences[0]
    if len(occurrences) > 1:
        print(f"Found {len(occurrences)} occurrences of '{target_token}', using {'last' if use_last_occurrence else 'first'} at position {target_pos}")
    
    # Run forward pass with attention
    model.set_attn_implementation("eager")
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
    
    # Extract attention from the target position
    attns = outputs.attentions
    lyr = layer_index if layer_index >= 0 else len(attns) + layer_index
    layer = attns[lyr][0]
    
    # Get attention from target position
    attn_vec = layer[:, target_pos, :].mean(dim=0)
    
    # Filter for image tokens only
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(
            getattr(processor, "image_token", "<|image_pad|>")
        )
    img_mask = token_ids == image_token_id
    vision_attn = attn_vec[img_mask]
    
    # Reshape to grid
    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        return None
    
    _, h, w = grid_thw[0].tolist()
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    h_grid = max(1, h // merge_size)
    w_grid = max(1, w // merge_size)
    
    if vision_attn.numel() != h_grid * w_grid:
        # Fallback reshaping
        n = vision_attn.numel()
        import math
        g_h = int(math.sqrt(n))
        while g_h > 1 and n % g_h != 0:
            g_h -= 1
        g_w = n // g_h
        h_grid, w_grid = g_h, g_w
    
    heatmap = vision_attn.reshape(h_grid, w_grid)
    return heatmap.to(device)


def _generate_box_descriptions(
    candidates: List,
    image_size: Tuple[int, int]
) -> str:
    """Generate text descriptions of bounding boxes for injection."""
    if not candidates:
        return "\nNo clickable regions detected."
    
    box_descriptions = []
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for i, cand in enumerate(candidates[:min(len(labels), len(candidates))]):
        x, y = cand.center
        # Normalize to 0-1000 scale
        x_norm = int(x * 1000 / image_size[0])
        y_norm = int(y * 1000 / image_size[1])
        
        # Describe the region
        region_desc = _get_region_description(x / image_size[0], y / image_size[1])
        
        box_descriptions.append(
            f"[{labels[i]}] {region_desc} region at ({x_norm}, {y_norm})"
        )
    
    return "\n\nDetected clickable regions:\n" + "\n".join(box_descriptions) + \
           "\n\nSelect which region to click (respond with just the letter):"


def _get_region_description(x_rel: float, y_rel: float) -> str:
    """Get a text description of where a region is located."""
    vertical = "top" if y_rel < 0.33 else "middle" if y_rel < 0.67 else "bottom"
    horizontal = "left" if x_rel < 0.33 else "center" if x_rel < 0.67 else "right"
    
    if horizontal == "center":
        return vertical
    elif vertical == "middle":
        return horizontal
    else:
        return f"{vertical}-{horizontal}"


def single_pass_with_injection(
    model_id: str,
    image_path: str,
    user_instruction: str = "Click on the submit button",
    device: str = "cuda",
    save_visualization: bool = True,
    layer_index: int = -1,
    interpolation_mode: str = "bilinear",
    display_width: int = 1920,
    display_height: int = 1080
) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    Perform single-pass attention-based clicking with bounding box injection.
    
    Args:
        model_id: Qwen3-VL model ID
        image_path: Path to UI screenshot
        user_instruction: What the user wants to click
        device: Compute device
        save_visualization: Whether to save annotated image
        layer_index: Which layer to extract attention from
        interpolation_mode: How to interpolate the heatmap
        
    Returns:
        (click_point, model_response) tuple
    """
    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load image
    print("Processing image...")
    image = Image.open(image_path).convert("RGB")
    
    # Build structured prompt with display dimensions
    prompt = _build_structured_prompt(user_instruction, display_width, display_height)
    
    # Prepare inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "file://placeholder"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[chat_text], images=[image], return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    
    # Move to device
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    
    # Extract attention at last occurrence of " action" token (with leading space)
    print("Extracting attention at last 'action' token...")
    heatmap = _extract_attention_at_token(
        model, processor, inputs, " action", layer_index, use_last_occurrence=True
    )
    
    if heatmap is None:
        print("Failed to extract attention")
        return None, "Failed to extract attention"
    
    print(f"Heatmap shape: {heatmap.shape}")
    
    # Generate bounding boxes from attention
    print("Generating bounding boxes...")
    targeter = ClickTargeter(
        threshold=0.5,
        min_area_ratio=0.001,
        max_candidates=5
    )
    
    candidates = targeter.generate_candidates(
        heatmap,
        image.size,
        adaptive_threshold=True,
        interpolation_mode=interpolation_mode
    )
    
    print(f"Found {len(candidates)} candidates")
    
    # Generate box descriptions for injection
    box_descriptions = _generate_box_descriptions(candidates, image.size)
    
    # Inject box descriptions into the input
    injected_text = chat_text + box_descriptions
    
    # Re-tokenize with injected content
    print("Injecting bounding boxes and continuing generation...")
    injected_inputs = processor(
        text=[injected_text], images=[image], return_tensors="pt"
    )
    injected_inputs.pop("token_type_ids", None)
    
    for k, v in list(injected_inputs.items()):
        if torch.is_tensor(v):
            injected_inputs[k] = v.to(device)
    
    # Generate with injected context
    with torch.inference_mode():
        output_ids = model.generate(
            **injected_inputs,
            max_new_tokens=32,
            do_sample=False,
            temperature=0.1,
        )
    
    # Decode response
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(injected_inputs["input_ids"], output_ids)
    ]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    
    print(f"Model response: {response}")
    
    # Parse the selected box
    selected_box = None
    click_point = None
    
    # Look for single letter response (A, B, C, etc.)
    match = re.search(r'\b([A-H])\b', response.upper())
    if match and candidates:
        letter = match.group(1)
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        if letter in labels:
            idx = labels.index(letter)
            if idx < len(candidates):
                selected_box = candidates[idx]
                click_point = selected_box.center
                print(f"Selected box {letter}: {click_point}")
    
    # Save visualization if requested
    if save_visualization and candidates:
        vis_image = visualize_with_labels(
            image, candidates, heatmap, selected_idx=labels.index(letter) if match else None,
            interpolation_mode=interpolation_mode
        )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem, _ = os.path.splitext(os.path.basename(image_path))
        output_path = f"{stem}_injection_{timestamp}.png"
        vis_image.save(output_path)
        print(f"Saved visualization to {output_path}")
    
    return click_point, response


def visualize_with_labels(
    image: Image.Image,
    candidates: List,
    heatmap: torch.Tensor,
    selected_idx: Optional[int] = None,
    interpolation_mode: str = "bilinear",
    alpha: float = 0.3
) -> Image.Image:
    """Visualize candidates with letter labels and highlight selected one."""
    # First apply heatmap overlay
    targeter = ClickTargeter()
    vis_image = targeter.visualize_candidates(
        image, [], heatmap, alpha=alpha, interpolation_mode=interpolation_mode
    )
    
    # Add labeled boxes
    draw = ImageDraw.Draw(vis_image)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    for i, cand in enumerate(candidates[:min(len(labels), len(candidates))]):
        x, y = cand.center
        bbox = cand.bbox
        
        # Draw box
        color = "green" if i == selected_idx else "red"
        width = 3 if i == selected_idx else 2
        draw.rectangle(
            [bbox[0], bbox[1], bbox[2], bbox[3]],
            outline=color,
            width=width
        )
        
        # Draw label
        label = labels[i]
        # Draw label background
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 20), label)
        draw.rectangle(text_bbox, fill="white")
        draw.text((bbox[0] + 2, bbox[1] - 18), label, fill=color)
        
        # Draw center point
        draw.ellipse(
            [x - 3, y - 3, x + 3, y + 3],
            fill=color
        )
    
    return vis_image


if __name__ == "__main__":
    # Hardcoded test setup from traj_0002 step 3
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    image_path = "/home/sdan/workspace/diffcua/differential_vision/agentnet_curated/traj_0002/step_0003_5d2f0d08-fb03-486e-9d7f-9340658a1376.png"
    user_instruction = "Click on the Save button for the healthy breakfast recipe"
    device = "cuda"
    layer_index = -1
    interpolation_mode = "bilinear"
    save_visualization = True
    display_width = 1920
    display_height = 1080

    click_point, response = single_pass_with_injection(
        model_id=model_id,
        image_path=image_path,
        user_instruction=user_instruction,
        device=device,
        save_visualization=save_visualization,
        layer_index=layer_index,
        interpolation_mode=interpolation_mode,
        display_width=display_width,
        display_height=display_height
    )

    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    if click_point:
        print(f"Click point: {click_point}")
    else:
        print("No click point determined")
    print(f"Model response: {response}")