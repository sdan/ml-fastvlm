"""
Example usage of attention-based click targeting.
"""

import torch
import os
from datetime import datetime
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from attention_extractor import AttentionExtractor
from click_targeter import ClickTargeter


def demo_attention_clicking(
    model_path: str,
    image_path: str,
    prompt: str = "Click on the button",
    device: str = "cuda",
    compare_prediction: bool = False,
    interpolation_mode: str = 'bilinear',
    save_raw_grid: bool = True
):
    """
    Demonstrate attention-based click targeting.

    Args:
        model_path: Path to LLaVA model
        image_path: Path to screenshot/UI image
        prompt: Instruction for the model
        device: Device to run on
        compare_prediction: Whether to decode and compare the model's own click prediction
        interpolation_mode: 'bilinear' for smooth heatmap, 'nearest' for blocky patches
        save_raw_grid: Whether to save the raw 16x16 attention grid
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
    
    # Prepare prompt
    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    
    # Extract attention heatmap (no re-encoding!)
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

    # Save raw 16x16 grid if requested
    if save_raw_grid:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem, _ = os.path.splitext(os.path.basename(image_path))
        grid_out = f"{stem}_attention_grid_{timestamp}.txt"
        with open(grid_out, 'w') as f:
            f.write(f"Raw {heatmap.shape} attention grid:\n\n")
            grid_np = heatmap.detach().cpu().numpy()
            for i in range(grid_np.shape[0]):
                for j in range(grid_np.shape[1]):
                    f.write(f"{grid_np[i, j]:.6f} ")
                f.write("\n")
        print(f"Saved raw attention grid to {grid_out}")

    # Generate click candidates
    print("Generating click candidates...")
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
    
    print(f"Found {len(candidates)} click candidates")

    # Always save a heatmap overlay (even if there are no candidates)
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem, _ = os.path.splitext(os.path.basename(image_path))
        heatmap_out = f"{stem}_heatmap_{timestamp}.png"
        overlay_only = targeter.visualize_candidates(
            image,
            candidates=[],  # no boxes; just heatmap overlay
            heatmap=heatmap,
            alpha=0.35,
            interpolation_mode=interpolation_mode
        )
        overlay_only.save(heatmap_out)
        print(f"Saved heatmap overlay to {heatmap_out}")
    except Exception as e:
        print(f"Failed to save heatmap overlay: {e}")
    
    predicted_click = None
    if compare_prediction:
        predicted_click = compare_with_model_prediction(
            model=model,
            tokenizer=tokenizer,
            image_tensor=image_tensor,
            image_size=image.size,
            input_ids=input_ids,
            device=device
        )
        if predicted_click:
            print(f"Decoded model click prediction: {predicted_click}")
        else:
            print("Model prediction did not include explicit click coordinates")
    
    # Select best click point
    if candidates:
        click_point = targeter.select_click_point(
            candidates,
            strategy='highest_attention'
        )
        print(f"Best click point: {click_point}")
        
        # Visualize results
        vis_image = targeter.visualize_candidates(
            image,
            candidates,
            heatmap,
            alpha=0.3,
            interpolation_mode=interpolation_mode
        )
        
        # Save visualization
        output_path = image_path.replace('.png', '_attention_clicks.png')
        vis_image.save(output_path)
        print(f"Saved visualization to {output_path}")
        
        return click_point, candidates, predicted_click
    
    print("No click candidates found")
    return None, [], predicted_click


def compare_with_model_prediction(
    model,
    tokenizer,
    image_tensor,
    image_size,
    input_ids,
    device
):
    """
    Compare attention-based click with model's actual prediction.
    """
    # Get model's prediction
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False
        )
    
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Model prediction: {prediction}")
    
    # Parse click coordinates if present
    # This would depend on the model's output format
    # Example: "click(123, 456)"
    import re
    match = re.search(r'click\((\d+),\s*(\d+)\)', prediction)
    if match:
        predicted_x, predicted_y = int(match.group(1)), int(match.group(2))
        print(f"Model predicted click: ({predicted_x}, {predicted_y})")
        return (predicted_x, predicted_y)
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Click on the most important button")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--compare-model",
        action="store_true",
        help="Decode model output and compare with attention-based click"
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest"],
        help="Interpolation mode: 'bilinear' for smooth, 'nearest' for blocky"
    )
    parser.add_argument(
        "--no-save-grid",
        action="store_true",
        help="Don't save the raw 16x16 attention grid"
    )

    args = parser.parse_args()
    
    result = demo_attention_clicking(
        args.model_path,
        args.image,
        args.prompt,
        args.device,
        args.compare_model,
        args.interpolation,
        not args.no_save_grid
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
                print(f"  Candidate {i+1}: center={cand.center}, score={cand.attention_score:.3f}")
        
        if args.compare_model:
            if predicted_click:
                print(f"  Model predicted click: {predicted_click}")
            else:
                print("  Model did not produce a parseable click prediction")
