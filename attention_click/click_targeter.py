"""
Generate and rank click targets from attention heatmaps.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy import ndimage
from PIL import Image, ImageDraw


@dataclass
class ClickCandidate:
    """A potential click target derived from attention."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    center: Tuple[int, int]  # (x, y) click point
    attention_score: float  # Average attention in this region
    area: int  # Pixel area
    rank: int  # Ranking among candidates


class ClickTargeter:
    """
    Generate click targets from attention heatmaps.
    """
    
    def __init__(self,
                 threshold: float = 0.5,
                 min_area_ratio: float = 0.001,
                 max_candidates: int = 10,
                 merge_overlap_threshold: float = 0.3):
        """
        Args:
            threshold: Attention threshold for candidate regions (0-1)
            min_area_ratio: Minimum region area as ratio of image area
            max_candidates: Maximum number of candidates to return
            merge_overlap_threshold: IoU threshold for merging overlapping boxes
        """
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.max_candidates = max_candidates
        self.merge_overlap_threshold = merge_overlap_threshold
    
    def generate_candidates(self,
                          heatmap: torch.Tensor,
                          image_size: Tuple[int, int],
                          adaptive_threshold: bool = True,
                          interpolation_mode: str = 'bilinear') -> List[ClickCandidate]:
        """
        Generate click candidates from attention heatmap.

        Args:
            heatmap: Attention heatmap [H, W] in patch coordinates
            image_size: Original image (width, height) in pixels
            adaptive_threshold: Use adaptive thresholding based on heatmap statistics
            interpolation_mode: 'bilinear' for smooth heatmap, 'nearest' for blocky patches

        Returns:
            List of ClickCandidate objects, ranked by attention score
        """
        if heatmap is None:
            return []
        
        # Normalize on GPU (stay on device)
        heat = heatmap.to(dtype=torch.float32)
        hmin = torch.min(heat)
        hmax = torch.max(heat)
        if (hmax - hmin) > 0:
            heat_norm = (heat - hmin) / (hmax - hmin)
        else:
            heat_norm = heat

        # Upscale to image resolution using GPU
        target_w, target_h = image_size
        heat_up = F.interpolate(
            heat_norm.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode=interpolation_mode,
            align_corners=False if interpolation_mode == 'bilinear' else None
        ).squeeze(0).squeeze(0)
        
        # Threshold to get binary mask
        if adaptive_threshold:
            mean = heat_up.mean()
            std = heat_up.std()
            thr = torch.clamp(mean + 1.0 * std, 0.3, 0.8)
        else:
            thr = torch.tensor(self.threshold, dtype=heat_up.dtype, device=heat_up.device)

        binary_mask_t = heat_up > thr

        # Move resized heatmap and mask to CPU numpy for connected components
        heatmap_resized = heat_up.detach().cpu().numpy().astype(np.float32, copy=False)
        binary_mask = binary_mask_t.detach().cpu().numpy()
        
        # Find connected components
        labeled, num_features = ndimage.label(binary_mask)
        
        if num_features == 0:
            return []
        
        # Extract bounding boxes
        candidates = []
        min_area = int(image_size[0] * image_size[1] * self.min_area_ratio)
        
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            
            # Skip small regions
            if component_mask.sum() < min_area:
                continue
            
            # Get bounding box
            rows, cols = np.where(component_mask)
            y1, y2 = rows.min(), rows.max() + 1
            x1, x2 = cols.min(), cols.max() + 1
            
            # Compute attention score for this region
            region_attention = heatmap_resized[component_mask].mean()
            
            # Create candidate
            candidate = ClickCandidate(
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                attention_score=float(region_attention),
                area=int(component_mask.sum()),
                rank=0  # Will be set after ranking
            )
            candidates.append(candidate)
        
        # Merge overlapping candidates
        candidates = self._merge_overlapping(candidates)
        
        # Rank by attention score
        candidates.sort(key=lambda c: c.attention_score, reverse=True)
        for i, candidate in enumerate(candidates):
            candidate.rank = i + 1
        
        # Return top candidates
        return candidates[:self.max_candidates]
    
    def visualize_candidates(self,
                            image: Image.Image,
                            candidates: List[ClickCandidate],
                            heatmap: Optional[torch.Tensor] = None,
                            alpha: float = 0.3,
                            interpolation_mode: str = 'bilinear') -> Image.Image:
        """
        Visualize click candidates on the image.

        Args:
            image: Original PIL image
            candidates: List of click candidates
            heatmap: Optional attention heatmap for overlay
            alpha: Transparency for heatmap overlay
            interpolation_mode: 'bilinear' for smooth heatmap, 'nearest' for blocky patches

        Returns:
            Image with visualized candidates
        """
        # Create a copy
        vis_image = image.copy()
        
        # Overlay heatmap if provided
        if heatmap is not None:
            # Normalize and resize on GPU for speed
            heat = heatmap.to(dtype=torch.float32)
            hmin = torch.min(heat)
            hmax = torch.max(heat)
            if (hmax - hmin) > 0:
                heat_norm = (heat - hmin) / (hmax - hmin)
            else:
                heat_norm = heat

            target_w, target_h = image.size
            heat_up = F.interpolate(
                heat_norm.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode=interpolation_mode,
                align_corners=False if interpolation_mode == 'bilinear' else None
            ).squeeze(0).squeeze(0)

            heatmap_resized = heat_up.detach().cpu().numpy().astype(np.float32, copy=False)
            
            # Create heatmap overlay
            heatmap_colored = self._colorize_heatmap(heatmap_resized)
            heatmap_image = Image.fromarray(heatmap_colored)
            
            # Blend with original
            vis_image = Image.blend(vis_image, heatmap_image, alpha)
        
        # Draw bounding boxes and centers
        draw = ImageDraw.Draw(vis_image)
        
        for candidate in candidates:
            # Draw bounding box
            color = self._rank_to_color(candidate.rank)
            draw.rectangle(candidate.bbox, outline=color, width=2)
            
            # Draw center point
            cx, cy = candidate.center
            draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=color, outline='white')
            
            # Draw rank label
            draw.text((candidate.bbox[0], candidate.bbox[1] - 15),
                     f"#{candidate.rank} ({candidate.attention_score:.2f})",
                     fill=color)
        
        return vis_image
    
    def select_click_point(self,
                          candidates: List[ClickCandidate],
                          strategy: str = 'highest_attention') -> Optional[Tuple[int, int]]:
        """
        Select the best click point from candidates.
        
        Args:
            candidates: List of click candidates
            strategy: Selection strategy:
                - 'highest_attention': Click on highest attention region
                - 'largest_area': Click on largest region
                - 'center_weighted': Balance attention and centrality
                
        Returns:
            (x, y) click coordinates or None if no candidates
        """
        if not candidates:
            return None
        
        if strategy == 'highest_attention':
            # Already sorted by attention
            return candidates[0].center
        
        elif strategy == 'largest_area':
            largest = max(candidates, key=lambda c: c.area)
            return largest.center
        
        elif strategy == 'center_weighted':
            # Prefer high attention near the image center to reduce corner bias.
            # Estimate the image size from candidate boxes (robust when image size is unknown).
            max_x = max(c.bbox[2] for c in candidates)
            max_y = max(c.bbox[3] for c in candidates)
            cx_img = max_x / 2.0
            cy_img = max_y / 2.0
            # Diagonal for normalization
            diag = (cx_img**2 + cy_img**2) ** 0.5 + 1e-6
            best = None
            best_score = -1.0
            for c in candidates:
                dx = c.center[0] - cx_img
                dy = c.center[1] - cy_img
                dist = (dx*dx + dy*dy) ** 0.5
                # Gaussian falloff toward edges; lambda controls penalty strength
                lam = 3.5
                weight = float(np.exp(-lam * (dist / diag) ** 2))
                score = c.attention_score * weight
                if score > best_score:
                    best_score = score
                    best = c
            return best.center if best is not None else candidates[0].center
        
        else:
            return candidates[0].center
    
    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1] range."""
        if heatmap.max() > heatmap.min():
            return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap
    
    def _resize_heatmap(self, 
                       heatmap: np.ndarray,
                       target_size: Tuple[int, int]) -> np.ndarray:
        """Resize heatmap to target size using bilinear interpolation."""
        from scipy.ndimage import zoom
        
        # Ensure dtype supported by scipy (float32)
        if heatmap.dtype != np.float32:
            heatmap = heatmap.astype(np.float32, copy=False)

        h, w = heatmap.shape
        target_w, target_h = target_size
        
        zoom_factors = (target_h / h, target_w / w)
        return zoom(heatmap, zoom_factors, order=1)  # Bilinear
    
    def _compute_adaptive_threshold(self, heatmap: np.ndarray) -> float:
        """Compute adaptive threshold based on heatmap statistics."""
        mean = heatmap.mean()
        std = heatmap.std()
        
        # Use mean + k*std as threshold
        # k can be tuned based on desired sensitivity
        k = 1.0
        threshold = mean + k * std
        
        # Clip to reasonable range
        return np.clip(threshold, 0.3, 0.8)
    
    def _merge_overlapping(self, candidates: List[ClickCandidate]) -> List[ClickCandidate]:
        """Merge overlapping bounding boxes."""
        if len(candidates) <= 1:
            return candidates
        
        merged = []
        used = set()
        
        for i, cand1 in enumerate(candidates):
            if i in used:
                continue
            
            # Find all overlapping candidates
            group = [cand1]
            used.add(i)
            
            for j, cand2 in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                
                if self._compute_iou(cand1.bbox, cand2.bbox) > self.merge_overlap_threshold:
                    group.append(cand2)
                    used.add(j)
            
            # Merge the group
            if len(group) == 1:
                merged.append(cand1)
            else:
                merged_candidate = self._merge_group(group)
                merged.append(merged_candidate)
        
        return merged
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_group(self, group: List[ClickCandidate]) -> ClickCandidate:
        """Merge a group of candidates into one."""
        # Take union of bounding boxes
        x1 = min(c.bbox[0] for c in group)
        y1 = min(c.bbox[1] for c in group)
        x2 = max(c.bbox[2] for c in group)
        y2 = max(c.bbox[3] for c in group)
        
        # Average attention scores
        avg_attention = np.mean([c.attention_score for c in group])
        
        # Sum areas
        total_area = sum(c.area for c in group)
        
        return ClickCandidate(
            bbox=(x1, y1, x2, y2),
            center=((x1 + x2) // 2, (y1 + y2) // 2),
            attention_score=avg_attention,
            area=total_area,
            rank=0
        )
    
    def _colorize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Convert grayscale heatmap to RGB."""
        # Simple red colormap
        colored = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
        colored[:, :, 0] = (heatmap * 255).astype(np.uint8)  # Red channel
        return colored
    
    def _rank_to_color(self, rank: int) -> str:
        """Map rank to color for visualization."""
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        return colors[min(rank - 1, len(colors) - 1)]
