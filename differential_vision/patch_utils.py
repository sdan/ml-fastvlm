"""
Utilities for patch-based image processing and change detection.
"""

import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional


def extract_patch(
    image: np.ndarray, 
    row: int, 
    col: int, 
    patch_size: int = 24
) -> np.ndarray:
    """Extract a patch from an image at the given grid position.
    
    Args:
        image: Input image as numpy array [H, W, C]
        row: Row index in the patch grid
        col: Column index in the patch grid
        patch_size: Size of each patch (default: 24 for CLIP)
    
    Returns:
        Extracted patch [patch_size, patch_size, C]
    """
    h_start = row * patch_size
    w_start = col * patch_size
    h_end = min(h_start + patch_size, image.shape[0])
    w_end = min(w_start + patch_size, image.shape[1])
    
    patch = image[h_start:h_end, w_start:w_end]
    
    # Pad if necessary (for edge patches)
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
        padded[:patch.shape[0], :patch.shape[1]] = patch
        patch = padded
    
    return patch


def extract_patch_window(
    image: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
    radius: int = 1,
) -> np.ndarray:
    """Extract a square window of patches centered at (row, col).

    The window spans (2*radius+1) patches per side. Edges are clamped and
    padded with zeros as needed to preserve a consistent output size.

    Args:
        image: Input image as numpy array [H, W, C]
        row: Center row index in the patch grid
        col: Center column index in the patch grid
        patch_size: Size of each patch (pixels)
        radius: Window radius in patches (default: 1 => 3x3 window)

    Returns:
        Extracted window [window_px, window_px, C] where window_px = patch_size*(2*radius+1)
    """
    window_patches = 2 * radius + 1
    window_px = window_patches * patch_size

    # Desired top-left in pixel coordinates
    h_start = (row - radius) * patch_size
    w_start = (col - radius) * patch_size
    h_end = (row + radius + 1) * patch_size
    w_end = (col + radius + 1) * patch_size

    # Clamp to image bounds
    src_h0 = max(h_start, 0)
    src_w0 = max(w_start, 0)
    src_h1 = min(h_end, image.shape[0])
    src_w1 = min(w_end, image.shape[1])

    # Compute destination placement within the fixed-size window
    dst_h0 = src_h0 - h_start
    dst_w0 = src_w0 - w_start
    dst_h1 = dst_h0 + (src_h1 - src_h0)
    dst_w1 = dst_w0 + (src_w1 - src_w0)

    out = np.zeros((window_px, window_px, image.shape[2]), dtype=image.dtype)
    if src_h1 > src_h0 and src_w1 > src_w0:
        out[dst_h0:dst_h1, dst_w0:dst_w1] = image[src_h0:src_h1, src_w0:src_w1]
    return out


def compute_patch_diff(
    img1: np.ndarray,
    img2: np.ndarray,
    patch_size: int = 24,
    threshold: float = 0.05
) -> np.ndarray:
    """Compute which patches have changed between two images.

    Vectorized implementation that handles edge patches without bias by normalizing
    with the valid pixel count per patch.

    Args:
        img1: First image [H, W, C] or [H, W]
        img2: Second image [H, W, C] or [H, W]
        patch_size: Size of patches to compare
        threshold: Change threshold (0-1), patches with mean diff above this are marked as changed

    Returns:
        Boolean mask [grid_h, grid_w] where True indicates changed patches
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    # Compute per-pixel absolute difference
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))

    # Average across color channels if present
    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    # Normalize to [0, 1] if needed
    if diff.size > 0 and diff.max() > 1.0:
        diff = diff / 255.0

    h, w = diff.shape
    grid_h = (h + patch_size - 1) // patch_size
    grid_w = (w + patch_size - 1) // patch_size

    # Pad to full patches
    pad_h = grid_h * patch_size - h
    pad_w = grid_w * patch_size - w
    if pad_h or pad_w:
        diff_padded = np.pad(diff, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        mask_padded = np.pad(np.ones_like(diff, dtype=np.float32), ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
    else:
        diff_padded = diff
        mask_padded = np.ones_like(diff, dtype=np.float32)

    # Reshape to blocks and compute mean normalized by valid counts
    diff_blocks = diff_padded.reshape(grid_h, patch_size, grid_w, patch_size)
    mask_blocks = mask_padded.reshape(grid_h, patch_size, grid_w, patch_size)
    sum_per_patch = diff_blocks.sum(axis=(1, 3))
    cnt_per_patch = mask_blocks.sum(axis=(1, 3)) + 1e-8
    patch_mean = sum_per_patch / cnt_per_patch

    return patch_mean > threshold


def compute_patch_diff_values(
    img1: np.ndarray,
    img2: np.ndarray,
    patch_size: int = 24,
    normalize: bool = True,
) -> np.ndarray:
    """Compute per-patch mean absolute difference values.

    Same computation as ``compute_patch_diff`` but returns the continuous
    per-patch mean difference (float32) instead of a boolean mask. Useful for
    ranking patches by change magnitude and building heatmaps.

    Args:
        img1: First image [H, W, C] or [H, W]
        img2: Second image [H, W, C] or [H, W]
        patch_size: Patch size in pixels
        normalize: If True, interpret pixel values in [0,255] and scale to [0,1]

    Returns:
        Float map of shape [grid_h, grid_w] with mean absolute diff per patch.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    if normalize and diff.size > 0 and diff.max() > 1.0:
        diff = diff / 255.0

    h, w = diff.shape
    grid_h = (h + patch_size - 1) // patch_size
    grid_w = (w + patch_size - 1) // patch_size

    pad_h = grid_h * patch_size - h
    pad_w = grid_w * patch_size - w
    if pad_h or pad_w:
        diff_padded = np.pad(diff, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        mask_padded = np.pad(np.ones_like(diff, dtype=np.float32), ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
    else:
        diff_padded = diff
        mask_padded = np.ones_like(diff, dtype=np.float32)

    diff_blocks = diff_padded.reshape(grid_h, patch_size, grid_w, patch_size)
    mask_blocks = mask_padded.reshape(grid_h, patch_size, grid_w, patch_size)
    sum_per_patch = diff_blocks.sum(axis=(1, 3))
    cnt_per_patch = mask_blocks.sum(axis=(1, 3)) + 1e-8
    patch_mean = (sum_per_patch / cnt_per_patch).astype(np.float32)
    return patch_mean


def overlay_patch_rects(
    image: np.ndarray,
    coords: list[tuple[int, int]],
    patch_size: int = 24,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw rectangle outlines around selected patch grid coordinates.

    Args:
        image: Input image [H, W, C], dtype uint8 or float in [0,1]
        coords: List of (row, col) patch coordinates
        patch_size: Patch size in pixels
        color: RGB color for rectangle outlines
        thickness: Outline thickness in pixels

    Returns:
        A copy of the image with rectangles overlaid (uint8).
    """
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8).copy()
    else:
        img = image.astype(np.uint8).copy()

    H, W = img.shape[:2]
    for (r, c) in coords:
        y0 = max(r * patch_size, 0)
        x0 = max(c * patch_size, 0)
        y1 = min((r + 1) * patch_size, H)
        x1 = min((c + 1) * patch_size, W)
        if y0 >= y1 or x0 >= x1:
            continue
        t = max(int(thickness), 1)
        # Top / Bottom
        img[y0:y0 + t, x0:x1] = color
        img[y1 - t:y1, x0:x1] = color
        # Left / Right
        img[y0:y1, x0:x0 + t] = color
        img[y0:y1, x1 - t:x1] = color

    return img


def visualize_changed_patches(
    image: np.ndarray,
    changed_mask: np.ndarray,
    patch_size: int = 24,
    highlight_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.3
) -> np.ndarray:
    """Visualize which patches have changed by overlaying a highlight.
    
    Args:
        image: Original image [H, W, C]
        changed_mask: Boolean mask of changed patches [grid_h, grid_w]
        patch_size: Size of each patch
        highlight_color: RGB color for highlighting changed patches
        alpha: Transparency of highlight overlay
    
    Returns:
        Visualized image with changed patches highlighted
    """
    vis_img = image.copy()
    if vis_img.max() <= 1.0:
        vis_img = (vis_img * 255).astype(np.uint8)
    
    overlay = np.zeros_like(vis_img)
    
    grid_h, grid_w = changed_mask.shape
    
    for i in range(grid_h):
        for j in range(grid_w):
            if changed_mask[i, j]:
                h_start = i * patch_size
                w_start = j * patch_size
                h_end = min(h_start + patch_size, image.shape[0])
                w_end = min(w_start + patch_size, image.shape[1])
                
                overlay[h_start:h_end, w_start:w_end] = highlight_color
    
    # Blend overlay with original image
    result = vis_img.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha
    return result.astype(np.uint8)


def compute_token_positions(
    grid_h: int,
    grid_w: int,
    vision_config: Optional[dict] = None
) -> torch.Tensor:
    """Compute position embeddings for vision tokens in a grid.
    
    Args:
        grid_h: Height of token grid
        grid_w: Width of token grid
        vision_config: Optional vision model config with position embedding info
    
    Returns:
        Position indices tensor [grid_h * grid_w]
    """
    # Simple row-major ordering
    positions = torch.arange(grid_h * grid_w)
    return positions.view(grid_h, grid_w)
