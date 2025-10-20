"""
Differential Vision Tokens for Fast Computer-Use Models

This module implements differential vision encoding to speed up sequential screenshot processing
by caching and reusing unchanged vision tokens between frames.
"""

from .differential_encoder import DifferentialVisionEncoder
from .patch_utils import extract_patch, compute_patch_diff

__all__ = ["DifferentialVisionEncoder", "extract_patch", "compute_patch_diff"]