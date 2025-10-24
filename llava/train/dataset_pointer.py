import copy
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import transformers

from llava import conversation as conversation_lib
from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)
from llava.train.train import preprocess, preprocess_multimodal
from llava.mm_utils import process_anyres_image


# Patterns to locate pointer coordinates inside assistant text
# We support several common formats from GUI-Actor-style traces and code snippets.
ACTION_PATTENS_XY = [
    # key=value pairs
    r"x=([-0-9.]+),\s*y=([-0-9.]+)",
    # function-style calls
    r"click\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)",
    r"pyautogui\.click\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)",
    r"pyautogui\.moveTo\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)",
    # bbox / from-to pairs (treated as two targets)
    r"from_coord=\[([-0-9.]+),\s*([-0-9.]+)\],\s*to_coord=\[([-0-9.]+),\s*([-0-9.]+)\]",
]


def _reformat_coordinates(text: str) -> Tuple[str, List[Tuple[float, float]]]:
    epsilon = 1e-3

    def adj(c: float) -> float:
        if abs(c) < epsilon:
            return epsilon
        if abs(c - 1) < epsilon:
            return 1 - epsilon
        return c

    all_matches = []
    for pattern in ACTION_PATTENS_XY:
        matches = list(re.finditer(pattern, text))
        for m in matches:
            all_matches.append((m.start(), m.groups()))

        # Replace occurrences with pointer tokens; choose 1 or 2 placeholders based on group count
        def _repl_fn(m):
            g = m.groups()
            if len(g) == 2:
                return f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
            elif len(g) == 4:
                return (
                    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}, "
                    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
                )
            return m.group(0)

        text = re.sub(pattern, _repl_fn, text)

    coordinates: List[Tuple[float, float]] = []
    all_matches.sort(key=lambda x: x[0])
    for _, groups in all_matches:
        if len(groups) == 2:
            x_str, y_str = groups
            x = adj(float(x_str))
            y = adj(float(y_str))
            coordinates.append((x, y))
        elif len(groups) == 4:
            x1_str, y1_str, x2_str, y2_str = groups
            coordinates.append((adj(float(x1_str)), adj(float(y1_str))))
            coordinates.append((adj(float(x2_str)), adj(float(y2_str))))
    return text, coordinates


@dataclass
class PointerDataArgs:
    image_folder: Optional[List[str]] = None
    is_multimodal: bool = True
    image_aspect_ratio: str = "square"
    mm_use_im_start_end: bool = False
    # Use a generic type to avoid import-time attribute issues across transformers versions
    image_processor: Optional[Any] = None
    image_grid_pinpoints: Optional[str] = None


class PointerSupervisedDataset(Dataset):
    """Dataset that injects pointer tokens and returns coordinates for supervision.

    Expected input JSON entries follow the usual LLaVA SFT format, optionally augmented with
    an assistant field 'bbox_gt' (normalized [x1, y1, x2, y2]) if multi-patch labels are desired.
    """

    def __init__(self, data_paths: List[str], tokenizer: transformers.PreTrainedTokenizer, data_args: PointerDataArgs):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict: List[Dict] = []

        for i, p in enumerate(data_paths):
            with open(p, "r") as f:
                data = json.load(f)
            # Tag each entry with which image_folder index to use
            data = [{**row, "img_path_idx": min(i, len(data_args.image_folder) - 1) if data_args.image_folder else 0} for row in data]
            self.list_data_dict.extend(data)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        # approximate for grouped sampling
        arr = []
        for sample in self.list_data_dict:
            txt_len = sum(len(conv["value"].split()) for conv in sample["conversations"]) if "conversations" in sample else 0
            arr.append(txt_len + (128 if "image" in sample else 0))
        return arr

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]

        # Standard LLaVA path: single image
        assert "image" in source, "Pointer dataset expects image-grounded samples"
        image_file = source["image"]
        img_path_idx = source.get("img_path_idx", 0)
        image_folder = self.data_args.image_folder[img_path_idx] if self.data_args.image_folder else ""
        image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        image_size = image.size

        # Clone and rewrite assistant text to include pointer tokens, capturing coordinates
        conversations = copy.deepcopy(source["conversations"])  # list of dicts with from/value
        coords_accum: List[Tuple[float, float]] = []
        for turn in conversations:
            if turn.get("from", "").lower() == "gpt":
                new_text, coords = _reformat_coordinates(turn["value"])
                turn["value"] = new_text
                coords_accum.extend(coords)

        # Preprocess (tokenize, mask) using the standard pipeline
        # Insert DEFAULT_IMAGE_TOKEN into the first human turn if not present
        has_img_token = any(DEFAULT_IMAGE_TOKEN in t["value"] for t in conversations)
        if not has_img_token:
            # Prepend the image token to the first user message
            for turn in conversations:
                if turn.get("from", "").lower() == "human":
                    turn["value"] = DEFAULT_IMAGE_TOKEN + "\n" + turn["value"].strip()
                    break

        sources = [conversations]
        sources = preprocess_multimodal(sources, self.data_args)
        proc = preprocess(sources, self.tokenizer, has_image=True)
        input_ids = proc["input_ids"][0]
        labels = proc["labels"][0]

        # Image processing consistent with LLaVA vision tower
        processor = self.data_args.image_processor
        if processor is None:
            raise ValueError("image_processor must be set in data_args")
        # AnyRes: convert to patches tensor [N, C, H, W]; otherwise a single CHW image tensor
        if self.data_args.image_aspect_ratio == "anyres" and self.data_args.image_grid_pinpoints is not None:
            image_tensor = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
        else:
            image_tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor,
            "image_sizes": torch.tensor(list(image_size), dtype=torch.long),
        }

        # Optional pointer supervision payloads
        out["coordinates"] = coords_accum if len(coords_accum) > 0 else None
        # If a bbox is available on assistant turn, include it for multi-patch masks on the model side
        # (the model can translate to a patch grid using vision tower config)
        bbox = None
        for turn in source["conversations"]:
            if turn.get("from", "").lower() == "gpt" and "bbox_gt" in turn:
                bbox = turn["bbox_gt"]
                break
        out["bboxes"] = bbox
        
        return out


@dataclass
class PointerCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Support AnyRes by passing a list of 4D tensors when present
        images_list = [b["images"] for b in batch]
        if any(img.ndim == 4 for img in images_list):
            images = images_list  # list of [N, C, H, W]
        else:
            images = torch.stack(images_list, dim=0)  # [B, C, H, W]
        image_sizes_tensor = torch.stack([b["image_sizes"] for b in batch], dim=0)
        # Convert to list of [W, H] for anyres utilities compatibility
        image_sizes = [s.tolist() for s in image_sizes_tensor]

        # Coordinates and optional masks are kept as Python lists to keep ragged shapes
        coords = [b.get("coordinates", None) for b in batch]
        bboxes = [b.get("bboxes", None) for b in batch]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "images": images,
            "image_sizes": image_sizes,
            "coordinates": coords,
            "bboxes": bboxes,
        }
