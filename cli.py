"""
Advanced CLI for LLaVA with optional features.
Supports differential vision encoding and attention-based click targeting.
"""

import os
import argparse
import torch
from PIL import Image
from typing import Optional, List, Tuple

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class AdvancedPredictor:
    """Advanced predictor with optional differential vision and attention clicking."""
    
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self.diff_encoder = None
        self.attention_extractor = None
        self.click_targeter = None
        
        self._setup_device()
        self._load_model()
        self._setup_optional_features()
    
    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
    
    def _load_model(self):
        """Load the LLaVA model."""
        model_path = os.path.expanduser(self.args.model_path)
        
        # Handle generation config
        self.generation_config = None
        gen_config_path = os.path.join(model_path, 'generation_config.json')
        if os.path.exists(gen_config_path):
            self.generation_config = os.path.join(model_path, '.generation_config.json')
            os.rename(gen_config_path, self.generation_config)
        
        # Load model
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, self.args.model_base, model_name, device=self.device
        )
        
        # Set pad token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    
    def _setup_optional_features(self):
        """Setup optional differential vision and attention clicking."""
        
        # Setup differential vision if requested
        if self.args.use_differential:
            from differential_vision import DifferentialVisionEncoder
            
            self.diff_encoder = DifferentialVisionEncoder(
                self.model,
                patch_size=self.args.diff_patch_size,
                diff_threshold=self.args.diff_threshold,
                max_changed_patches=self.args.diff_max_patches,
                skip_small_updates=self.args.diff_skip_small,
                device=torch.device(self.device)
            )
            
            # Wrap encode_images
            self.original_encode = self.model.encode_images
            self.model.encode_images = self._differential_encode_wrapper
        
        # Setup attention clicking if requested
        if self.args.use_attention_click:
            from attention_click import AttentionExtractor, ClickTargeter
            
            self.attention_extractor = AttentionExtractor(self.model, self.tokenizer)
            self.click_targeter = ClickTargeter(
                threshold=self.args.click_threshold,
                min_area_ratio=self.args.click_min_area,
                max_candidates=self.args.click_max_candidates
            )
    
    def _differential_encode_wrapper(self, images, image_sizes=None, return_stats=False):
        """Wrapper for differential encoding."""
        try:
            return self.diff_encoder.encode(images, image_sizes=image_sizes, return_stats=return_stats)
        except Exception as e:
            print(f"Differential encoding failed: {e}, falling back to normal encoding")
            return self.original_encode(images)
    
    def process_single_image(self, image_path: str, prompt: str) -> dict:
        """Process a single image with optional features."""
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Get model dtype
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        image_tensor = image_tensor.to(device=self.device, dtype=model_dtype)
        
        # Prepare prompt
        qs = prompt
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        result = {
            'image_path': image_path,
            'prompt': prompt,
            'image_size': image.size
        }
        
        # Extract attention heatmap if enabled
        if self.args.use_attention_click:
            heatmap = self.attention_extractor.extract_attention_heatmap(
                input_ids,
                image_tensor,
                image.size,
                layer_index=self.args.attention_layer,
                query_strategy=self.args.attention_query
            )
            
            if heatmap is not None:
                # Generate click candidates
                candidates = self.click_targeter.generate_candidates(
                    heatmap,
                    image.size,
                    adaptive_threshold=True
                )
                
                if candidates:
                    click_point = self.click_targeter.select_click_point(
                        candidates,
                        strategy=self.args.click_strategy
                    )
                    result['click_point'] = click_point
                    result['click_candidates'] = [(c.center, c.attention_score) for c in candidates]
                    
                    # Save visualization if requested
                    if self.args.save_attention_viz:
                        vis_image = self.click_targeter.visualize_candidates(
                            image, candidates, heatmap, alpha=0.3
                        )
                        viz_path = image_path.replace('.png', '_attention.png').replace('.jpg', '_attention.jpg')
                        vis_image.save(viz_path)
                        result['visualization'] = viz_path
        
        # Generate text response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=self.args.temperature > 0,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                num_beams=self.args.num_beams,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True
            )
            
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            result['response'] = output_text
        
        # Get differential stats if enabled
        if self.args.use_differential and hasattr(self.diff_encoder, 'get_stats'):
            result['diff_stats'] = self.diff_encoder.get_stats()
        
        return result
    
    def process_sequence(self, image_paths: List[str], prompt: str) -> List[dict]:
        """Process a sequence of images with differential encoding."""
        results = []
        
        # Reset differential encoder cache if enabled
        if self.diff_encoder:
            self.diff_encoder.reset_cache()
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing frame {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.process_single_image(image_path, prompt)
            results.append(result)
        
        return results
    
    def cleanup(self):
        """Cleanup and restore model state."""
        # Restore original encoding if wrapped
        if self.diff_encoder and hasattr(self, 'original_encode'):
            self.model.encode_images = self.original_encode
        
        # Restore generation config
        if self.generation_config is not None:
            model_path = os.path.expanduser(self.args.model_path)
            os.rename(self.generation_config, os.path.join(model_path, 'generation_config.json'))


def main():
    parser = argparse.ArgumentParser(description="Advanced LLaVA CLI with optional features")
    
    # Basic arguments
    parser.add_argument("--model-path", type=str, default="./llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Describe the image.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Single image file")
    input_group.add_argument("--image-dir", type=str, help="Directory of images (for sequence)")
    input_group.add_argument("--image-list", type=str, nargs="+", help="List of image files")
    
    # Differential vision options
    diff_group = parser.add_argument_group("Differential Vision")
    diff_group.add_argument("--use-differential", action="store_true", 
                            help="Enable differential vision encoding for sequences")
    diff_group.add_argument("--diff-threshold", type=float, default=0.05,
                            help="Patch change threshold (0-1)")
    diff_group.add_argument("--diff-max-patches", type=int, default=50,
                            help="Max changed patches before full re-encode")
    diff_group.add_argument("--diff-patch-size", type=int, default=None,
                            help="Override vision patch size")
    diff_group.add_argument("--diff-skip-small", action="store_true",
                            help="Skip re-encoding for small changes")
    
    # Attention clicking options
    attn_group = parser.add_argument_group("Attention-Based Clicking")
    attn_group.add_argument("--use-attention-click", action="store_true",
                            help="Enable attention-based click targeting")
    attn_group.add_argument("--attention-layer", type=int, default=-1,
                            help="Which layer to extract attention from")
    attn_group.add_argument("--attention-query", type=str, default="last",
                            choices=["last", "last_text", "cls"],
                            help="Query token selection strategy")
    attn_group.add_argument("--click-threshold", type=float, default=0.5,
                            help="Attention threshold for click regions")
    attn_group.add_argument("--click-min-area", type=float, default=0.001,
                            help="Minimum click region area ratio")
    attn_group.add_argument("--click-max-candidates", type=int, default=5,
                            help="Maximum click candidates to generate")
    attn_group.add_argument("--click-strategy", type=str, default="highest_attention",
                            choices=["highest_attention", "largest_area", "center_weighted"],
                            help="Click point selection strategy")
    attn_group.add_argument("--save-attention-viz", action="store_true",
                            help="Save attention visualization images")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output-json", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AdvancedPredictor(args)
    
    try:
        # Determine input images
        if args.image:
            # Single image
            result = predictor.process_single_image(args.image, args.prompt)
            
            # Print results
            print(f"\nResponse: {result['response']}")
            
            if 'click_point' in result:
                print(f"Suggested click: {result['click_point']}")
                if args.verbose and 'click_candidates' in result:
                    print("All candidates:")
                    for i, (point, score) in enumerate(result['click_candidates'][:3]):
                        print(f"  {i+1}. {point} (score: {score:.3f})")
            
            if 'diff_stats' in result and args.verbose:
                print(f"Differential stats: {result['diff_stats']}")
        
        else:
            # Multiple images
            if args.image_dir:
                import glob
                patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
                image_paths = []
                for pattern in patterns:
                    image_paths.extend(glob.glob(os.path.join(args.image_dir, pattern)))
                image_paths = sorted(image_paths)
            else:
                image_paths = args.image_list
            
            if not image_paths:
                print("No images found!")
                return
            
            # Process sequence
            results = predictor.process_sequence(image_paths, args.prompt)
            
            # Print results
            for i, result in enumerate(results):
                print(f"\n[Frame {i+1}] {os.path.basename(result['image_path'])}")
                print(f"Response: {result['response']}")
                
                if 'click_point' in result:
                    print(f"Click: {result['click_point']}")
            
            # Print differential stats if enabled
            if args.use_differential and args.verbose:
                total_stats = {}
                for result in results:
                    if 'diff_stats' in result:
                        for key, value in result['diff_stats'].items():
                            if key not in total_stats:
                                total_stats[key] = []
                            total_stats[key].append(value)
                
                if total_stats:
                    print("\n=== Differential Encoding Summary ===")
                    for key, values in total_stats.items():
                        if isinstance(values[0], (int, float)):
                            avg = sum(values) / len(values)
                            print(f"{key}: avg={avg:.2f}, total={sum(values)}")
        
        # Save JSON output if requested
        if args.output_json:
            import json
            output_data = result if args.image else results
            with open(args.output_json, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults saved to {args.output_json}")
    
    finally:
        predictor.cleanup()


if __name__ == "__main__":
    main()