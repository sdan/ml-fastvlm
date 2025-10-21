"""
Attention-based click targeting for computer-use agents.

This module provides functionality to:
1. Extract attention heatmaps from vision-language models
2. Generate bounding boxes from high-attention regions
3. Rank and select click targets based on attention patterns

Independent of differential vision encoding - works with any VLM.
"""

from .attention_extractor import AttentionExtractor
from .click_targeter import ClickTargeter

__all__ = ['AttentionExtractor', 'ClickTargeter']