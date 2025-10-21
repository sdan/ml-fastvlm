"""
Efficient attention extraction from vision-language models.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


class AttentionExtractor:
    """
    Extracts attention heatmaps from VLMs without re-encoding images.
    
    Optimized for LLaVA but extensible to other architectures.
    """
    
    def __init__(self, model, tokenizer):
        """
        Args:
            model: The vision-language model (e.g., LLaVA)
            tokenizer: The tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self._cache_model_info()
    
    def _cache_model_info(self):
        """Cache frequently accessed model attributes."""
        self.vision_tower = self.model.get_model().get_vision_tower()
        self.num_patches_per_side = getattr(self.vision_tower, 'num_patches_per_side', None)

        # Debug: Print vision tower info
        print(f"DEBUG: Vision tower patch grid: {self.num_patches_per_side}x{self.num_patches_per_side} = {self.num_patches_per_side**2 if self.num_patches_per_side else 'unknown'} patches")
        if hasattr(self.vision_tower.config, 'image_size'):
            print(f"DEBUG: Vision tower image size: {self.vision_tower.config.image_size}")
        if hasattr(self.vision_tower.config, 'patch_size'):
            print(f"DEBUG: Vision tower patch size: {self.vision_tower.config.patch_size}")

        # Cache image token index
        from llava.constants import IMAGE_TOKEN_INDEX
        self.image_token_index = IMAGE_TOKEN_INDEX

        # Determine if model uses anyres
        self.image_aspect_ratio = getattr(self.model.config, 'image_aspect_ratio', 'square')
        self.mm_patch_merge_type = getattr(self.model.config, 'mm_patch_merge_type', 'flat')

        print(f"DEBUG: image_aspect_ratio={self.image_aspect_ratio}, mm_patch_merge_type={self.mm_patch_merge_type}")
    
    def get_vision_token_count(self, 
                               image_size: Tuple[int, int],
                               prompt_ids: Optional[torch.Tensor] = None,
                               valid_len: Optional[int] = None) -> int:
        """
        Efficiently determine the number of vision tokens without re-encoding.
        
        Args:
            image_size: (width, height) of the original image
            prompt_ids: Optional prompt token IDs for validation
            valid_len: Optional total sequence length for validation
            
        Returns:
            Number of vision tokens
        """
        # Fast path 1: Use cached patch grid for pad/flat mode
        if (self.image_aspect_ratio in ['pad', 'square', None] and 
            self.mm_patch_merge_type == 'flat' and 
            self.num_patches_per_side is not None):
            return self.num_patches_per_side * self.num_patches_per_side
        
        # Fast path 2: Derive from sequence length (single image)
        if prompt_ids is not None and valid_len is not None:
            # Count non-image tokens in prompt
            text_tokens = (prompt_ids[0] != self.image_token_index).sum().item()
            # Vision tokens = total - text
            return valid_len - text_tokens
        
        # Fast path 3: Compute anyres grid without encoding
        if self.image_aspect_ratio == 'anyres':
            from llava.mm_utils import get_anyres_image_grid_shape
            
            grid_pinpoints = getattr(self.model.config, 'image_grid_pinpoints', None)
            if grid_pinpoints:
                # Get patch size from vision tower
                if hasattr(self.vision_tower.config, 'patch_size'):
                    patch_size = self.vision_tower.config.patch_size
                elif hasattr(self.vision_tower.config, 'image_size'):
                    # Derive from image size and num patches
                    img_size = self.vision_tower.config.image_size
                    patch_size = img_size // self.num_patches_per_side if self.num_patches_per_side else 14
                else:
                    patch_size = 14  # Common default
                
                grid_w, grid_h = get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size)
                base_tokens = self.num_patches_per_side * self.num_patches_per_side if self.num_patches_per_side else 256
                
                # Add tile tokens
                num_tiles = grid_w * grid_h // (patch_size * patch_size)
                total = base_tokens + num_tiles * base_tokens
                
                # Add newline tokens if using unpad
                if 'unpad' in self.mm_patch_merge_type:
                    total += grid_h  # One newline per row
                
                return total
        
        # Fallback: Re-encode (should rarely happen)
        return self._fallback_encode_count(image_size)
    
    def _fallback_encode_count(self, image_size: Tuple[int, int]) -> int:
        """Fallback to encoding for token count (last resort)."""
        # This would require the actual image tensor
        # For now, return a reasonable default
        if self.num_patches_per_side:
            return self.num_patches_per_side * self.num_patches_per_side
        return 576  # 24x24 default
    
    def extract_attention_heatmap(self,
                                 prompt_ids: torch.Tensor,
                                 image_tensor: torch.Tensor,
                                 image_size: Tuple[int, int],
                                 layer_index: int = -1,
                                 query_strategy: str = 'last',
                                 return_raw: bool = False) -> Optional[torch.Tensor]:
        """
        Extract attention heatmap from model forward pass.
        
        Args:
            prompt_ids: Prompt token IDs with IMAGE_TOKEN_INDEX placeholders
            image_tensor: Preprocessed image tensor
            image_size: Original image (width, height)
            layer_index: Which transformer layer to use (-1 for last)
            query_strategy: How to select query token:
                - 'last': Last token in sequence
                - 'last_text': Last non-image token
                - 'mean_last_k': Average of last k text tokens
                - 'cls': First token (if available)
            return_raw: Return raw attention instead of reshaped heatmap
            
        Returns:
            Attention heatmap [H, W] or raw attention vector if return_raw=True
        """
        device = next(self.model.parameters()).device
        
        # Ensure attention output
        original_impl = self._force_eager_attention()
        
        try:
            # Build multimodal inputs
            with torch.inference_mode():
                (_, position_ids, attention_mask, _, inputs_embeds, _) = \
                    self.model.prepare_inputs_labels_for_multimodal(
                        prompt_ids, None, None, None, None,
                        image_tensor, image_sizes=[image_size]
                    )
            
            # Move to device
            inputs_embeds = inputs_embeds.to(device)
            if position_ids is not None:
                position_ids = position_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                valid_len = int(attention_mask[0].sum().item())
            else:
                valid_len = inputs_embeds.shape[1]
            
            # Forward with attention
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=True,
                    return_dict=True
                )
            
            if not outputs.attentions:
                return None
            
            # Extract attention from specified layer
            attentions = outputs.attentions
            layer_idx = layer_index if layer_index >= 0 else len(attentions) + layer_index
            layer_attn = attentions[layer_idx][0][:, :valid_len, :valid_len]

            # Debug: Check layer attention
            print(f"DEBUG: layer_attn shape={layer_attn.shape}, valid_len={valid_len}")
            print(f"DEBUG: layer_attn has NaN: {torch.isnan(layer_attn).any()}")

            # Check attention at different positions
            for test_pos in [valid_len-1, valid_len-2, valid_len-3]:
                if test_pos >= 0:
                    test_attn = layer_attn[:, test_pos, :].mean(dim=0)
                    print(f"DEBUG: pos {test_pos}: min={test_attn.min():.6f}, max={test_attn.max():.6f}, has_nan={torch.isnan(test_attn).any()}")

            # Select query position based on strategy
            query_pos = self._select_query_position(
                prompt_ids, valid_len, query_strategy, attention_mask
            )

            # WORKAROUND: If last position has NaN, use second-to-last
            test_attn = layer_attn[:, query_pos, :].mean(dim=0)
            if torch.isnan(test_attn).any() and query_pos == valid_len - 1:
                print(f"DEBUG: Last position has NaN attention, using second-to-last position")
                query_pos = valid_len - 2

            # Get attention from query to all positions
            attn_weights = layer_attn[:, query_pos, :].mean(dim=0)  # Average over heads

            # Debug: Check attention stats
            print(f"DEBUG: query_pos={query_pos}, valid_len={valid_len}")
            print(f"DEBUG: attn_weights shape={attn_weights.shape}")
            print(f"DEBUG: attn_weights min={attn_weights.min():.6f}, max={attn_weights.max():.6f}, mean={attn_weights.mean():.6f}")
            print(f"DEBUG: Has NaN: {torch.isnan(attn_weights).any()}, Has Inf: {torch.isinf(attn_weights).any()}")

            # Build image token mask efficiently
            num_vision_tokens = self.get_vision_token_count(
                image_size, prompt_ids, valid_len
            )
            mask = self._build_vision_mask(prompt_ids[0], num_vision_tokens, valid_len)

            if mask is None:
                return None

            # Debug: Check mask stats
            print(f"DEBUG: num_vision_tokens={num_vision_tokens}")
            print(f"DEBUG: mask sum={mask.sum().item()} (should match num_vision_tokens)")
            print(f"DEBUG: vision token positions: start={mask.nonzero()[0].item() if mask.any() else 'N/A'}")

            # Extract vision attention
            vision_attn = attn_weights[mask]

            # Debug: Check vision attention stats
            print(f"DEBUG: vision_attn shape={vision_attn.shape}")
            print(f"DEBUG: vision_attn min={vision_attn.min():.6f}, max={vision_attn.max():.6f}, mean={vision_attn.mean():.6f}")
            print(f"DEBUG: vision_attn Has NaN: {torch.isnan(vision_attn).any()}")
            
            if return_raw:
                return vision_attn
            
            # Reshape to 2D grid
            return self._reshape_to_grid(vision_attn, num_vision_tokens)
            
        finally:
            self._restore_attention_impl(original_impl)
    
    def _force_eager_attention(self) -> Optional[str]:
        """Force eager attention implementation for attention output."""
        original = None
        if hasattr(self.model, 'set_attn_implementation'):
            try:
                current = getattr(self.model.config, '_attn_implementation', None)
                if current != 'eager':
                    original = current
                    self.model.set_attn_implementation('eager')
            except:
                pass
        elif hasattr(self.model.config, '_attn_implementation'):
            current = self.model.config._attn_implementation
            if current != 'eager':
                original = current
                self.model.config._attn_implementation = 'eager'
        return original
    
    def _restore_attention_impl(self, original: Optional[str]):
        """Restore original attention implementation."""
        if original is not None:
            if hasattr(self.model, 'set_attn_implementation'):
                try:
                    self.model.set_attn_implementation(original)
                except:
                    pass
            elif hasattr(self.model.config, '_attn_implementation'):
                self.model.config._attn_implementation = original
    
    def _select_query_position(self, 
                              prompt_ids: torch.Tensor,
                              valid_len: int,
                              strategy: str,
                              attention_mask: Optional[torch.Tensor]) -> int:
        """Select query token position based on strategy."""
        if strategy == 'last':
            return valid_len - 1
        
        elif strategy == 'last_text':
            # Find last non-image token position
            # This requires tracking which positions are image tokens
            # For simplicity, use last position minus estimated image tokens at end
            return valid_len - 1  # TODO: Implement proper last text token finding
        
        elif strategy == 'cls':
            return 0
        
        elif strategy.startswith('mean_last_'):
            # Extract k from strategy (e.g., 'mean_last_5')
            try:
                k = int(strategy.split('_')[-1])
                # For now, just use last position
                # TODO: Implement averaging over last k positions
                return valid_len - 1
            except:
                return valid_len - 1
        
        else:
            return valid_len - 1
    
    def _build_vision_mask(self, 
                          prompt_tokens: torch.Tensor,
                          num_vision_tokens: int,
                          total_len: int) -> Optional[torch.Tensor]:
        """Build mask for vision token positions."""
        device = prompt_tokens.device
        mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        
        # Find IMAGE_TOKEN_INDEX position
        image_pos = (prompt_tokens == self.image_token_index).nonzero(as_tuple=False)
        
        if image_pos.numel() == 0:
            return None
        
        # Mark vision token positions
        start_pos = image_pos[0, 0].item()
        end_pos = min(start_pos + num_vision_tokens, total_len)
        mask[start_pos:end_pos] = True
        
        return mask
    
    def _reshape_to_grid(self, 
                        vision_attn: torch.Tensor,
                        num_tokens: int) -> Optional[torch.Tensor]:
        """Reshape flat attention to 2D grid."""
        if self.num_patches_per_side and num_tokens == self.num_patches_per_side ** 2:
            # Simple square grid
            side = self.num_patches_per_side
            return vision_attn.reshape(side, side)
        
        # Try to factor into rectangle
        for h in range(int(np.sqrt(num_tokens)), 0, -1):
            if num_tokens % h == 0:
                w = num_tokens // h
                return vision_attn.reshape(h, w)
        
        return None

        
