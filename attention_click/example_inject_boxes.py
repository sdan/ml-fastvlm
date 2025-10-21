"""
Example of injecting precomputed boxes back into the model for selection.
Instead of comparing centroids to model predictions, we:
1. Do a forward pass to compute candidate boxes
2. Inject those boxes back into the model
3. Have the model pick one
"""

import torch
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from attention_extractor import AttentionExtractor
from click_targeter import ClickTargeter


def compute_and_inject_boxes(
    model_path: str,
    image_path: str,
    prompt: str = "Click on the button",
    device: str = "cuda",
    interpolation_mode: str = 'bilinear',
    max_boxes: int = 5,
    visualize_boxes: bool = True
):
    """
    Compute candidate boxes from attention, then inject them back into the model
    for selection.
    
    Args:
        model_path: Path to LLaVA model
        image_path: Path to screenshot/UI image
        prompt: Original instruction for the model
        device: Device to run on
        interpolation_mode: 'bilinear' for smooth heatmap, 'nearest' for blocky patches
        max_boxes: Maximum number of boxes to present to the model
        visualize_boxes: Whether to save visualization of numbered boxes
    """
    # Load model
    print("Loading model...")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, "llava", device=device
    )
    
    # Load and process image
    print("Processing image...")
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # PHASE 1: Extract attention and compute candidate boxes
    print("\n=== PHASE 1: Computing candidate boxes ===")
    
    # Prepare initial prompt for attention extraction
    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    
    # Extract attention heatmap
    print("Extracting attention heatmap...")
    extractor = AttentionExtractor(model, tokenizer)
    heatmap = extractor.extract_attention_heatmap(
        input_ids,
        image_tensor,
        image.size,
        layer_index=-1,  # Last layer
        query_strategy='last'  # Last token attention
    )
    
    if heatmap is None:
        print("Failed to extract attention heatmap")
        return None
    
    print(f"Heatmap shape: {heatmap.shape}")
    
    # Generate click candidates (boxes)
    print("Generating candidate boxes...")
    targeter = ClickTargeter(
        threshold=0.5,
        min_area_ratio=0.001,
        max_candidates=max_boxes
    )
    
    candidates = targeter.generate_candidates(
        heatmap,
        image.size,
        adaptive_threshold=True,
        interpolation_mode=interpolation_mode
    )
    
    if not candidates:
        print("No candidate boxes found from attention")
        return None
    
    print(f"Found {len(candidates)} candidate boxes")
    
    # Limit to max_boxes
    candidates = candidates[:max_boxes]
    
    # PHASE 2: Inject boxes back into model for selection
    print(f"\n=== PHASE 2: Injecting {len(candidates)} boxes for model selection ===")
    
    # Create annotated image with numbered boxes if requested
    annotated_image = image.copy()
    if visualize_boxes:
        annotated_image = draw_numbered_boxes(annotated_image, candidates)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem, _ = os.path.splitext(os.path.basename(image_path))
        boxes_vis_path = f"{stem}_numbered_boxes_{timestamp}.png"
        annotated_image.save(boxes_vis_path)
        print(f"Saved numbered boxes visualization to {boxes_vis_path}")
    
    # Format boxes information for the model
    boxes_description = format_boxes_for_prompt(candidates)
    
    # Create new prompt with injected boxes
    injection_prompt = f"""{DEFAULT_IMAGE_TOKEN}

I have precomputed {len(candidates)} candidate regions for the task: "{prompt}"

The candidate regions are:
{boxes_description}

Please analyze these regions and select the most appropriate one for the task. Respond with:
1. The box number you select (e.g., "Box 1")
2. Brief explanation of why you chose it
3. The exact click coordinates from the center of your chosen box"""
    
    print("\nInjection prompt created with box descriptions")
    
    # Prepare new conversation with injected boxes
    conv_inject = conv_templates["qwen_2"].copy()
    conv_inject.append_message(conv_inject.roles[0], injection_prompt)
    conv_inject.append_message(conv_inject.roles[1], None)
    prompt_inject_text = conv_inject.get_prompt()
    
    input_ids_inject = tokenizer_image_token(
        prompt_inject_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    
    # Get model's selection
    print("Getting model's box selection...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids_inject,
            images=image_tensor,
            image_sizes=[image.size],
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nModel's response:\n{response}")
    
    # Parse the selected box
    selected_box, selected_candidate = parse_box_selection(response, candidates)
    
    if selected_box is not None:
        print(f"\n=== RESULT ===")
        print(f"Model selected: Box {selected_box}")
        print(f"Click coordinates: {selected_candidate.center}")
        print(f"Attention score: {selected_candidate.attention_score:.3f}")
        
        # Save final visualization with selected box highlighted
        final_vis = visualize_selection(
            image, 
            candidates, 
            selected_box - 1,  # Convert to 0-indexed
            heatmap,
            interpolation_mode
        )
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem, _ = os.path.splitext(os.path.basename(image_path))
        final_path = f"{stem}_selected_box_{timestamp}.png"
        final_vis.save(final_path)
        print(f"Saved final selection visualization to {final_path}")
        
        return selected_candidate.center, candidates, selected_box
    else:
        print("Could not parse box selection from model response")
        return None, candidates, None


def format_boxes_for_prompt(candidates):
    """Format candidate boxes as text description for the model."""
    descriptions = []
    for i, cand in enumerate(candidates, 1):
        x1, y1, x2, y2 = cand.bbox
        cx, cy = cand.center
        descriptions.append(
            f"Box {i}: Region from ({x1}, {y1}) to ({x2}, {y2}), "
            f"center at ({cx}, {cy}), attention score: {cand.attention_score:.3f}"
        )
    return "\n".join(descriptions)


def draw_numbered_boxes(image, candidates):
    """Draw numbered boxes on the image."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i, cand in enumerate(candidates, 1):
        x1, y1, x2, y2 = cand.bbox
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw number label with background
        label = str(i)
        bbox = draw.textbbox((x1, y1), label, font=font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        
        # Draw white background for label
        draw.rectangle(
            [x1, y1 - label_height - 4, x1 + label_width + 4, y1],
            fill="white"
        )
        
        # Draw label text
        draw.text((x1 + 2, y1 - label_height - 2), label, fill="red", font=font)
        
        # Draw center point
        cx, cy = cand.center
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill="blue", outline="white")
    
    return img_copy


def parse_box_selection(response, candidates):
    """Parse the model's box selection from its response."""
    import re
    
    # Look for "Box N" pattern
    match = re.search(r'Box\s+(\d+)', response, re.IGNORECASE)
    if match:
        box_num = int(match.group(1))
        if 1 <= box_num <= len(candidates):
            return box_num, candidates[box_num - 1]
    
    # Also try to find just a number at the beginning
    match = re.search(r'^(\d+)', response.strip())
    if match:
        box_num = int(match.group(1))
        if 1 <= box_num <= len(candidates):
            return box_num, candidates[box_num - 1]
    
    return None, None


def visualize_selection(image, candidates, selected_idx, heatmap, interpolation_mode):
    """Visualize the final selection with highlighted box."""
    from click_targeter import ClickTargeter
    
    # Create targeter for visualization
    targeter = ClickTargeter()
    
    # Create base visualization with heatmap
    vis_image = targeter.visualize_candidates(
        image,
        candidates,
        heatmap,
        alpha=0.3,
        interpolation_mode=interpolation_mode
    )
    
    # Highlight the selected box
    draw = ImageDraw.Draw(vis_image)
    
    if 0 <= selected_idx < len(candidates):
        selected = candidates[selected_idx]
        x1, y1, x2, y2 = selected.bbox
        
        # Draw thicker border for selected box
        for offset in range(3):
            draw.rectangle(
                [x1-offset, y1-offset, x2+offset, y2+offset],
                outline="lime",
                width=1
            )
        
        # Draw center with larger marker
        cx, cy = selected.center
        draw.ellipse([cx-5, cy-5, cx+5, cy+5], fill="lime", outline="black")
        
        # Add "SELECTED" label
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        label = "SELECTED"
        bbox = draw.textbbox((x1, y1), label, font=font)
        label_height = bbox[3] - bbox[1]
        draw.text((x1 + 5, y1 - label_height - 10), label, fill="lime", font=font)
    
    return vis_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Click on the most important button")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-boxes", type=int, default=5, help="Maximum boxes to present to model")
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest"],
        help="Interpolation mode: 'bilinear' for smooth, 'nearest' for blocky"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Don't save visualization of numbered boxes"
    )
    
    args = parser.parse_args()
    
    result = compute_and_inject_boxes(
        args.model_path,
        args.image,
        args.prompt,
        args.device,
        args.interpolation,
        args.max_boxes,
        not args.no_visualize
    )
    
    if result is not None:
        click_point, candidates, selected_box = result
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Total candidates generated: {len(candidates)}")
        print(f"Model selected: Box {selected_box}")
        print(f"Final click point: {click_point}")
    else:
        print("\nFailed to complete box injection process")