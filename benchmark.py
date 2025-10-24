"""
Benchmarking utilities for sequencing image predictions.

Provides a small helper to time per-frame inference and print a
per-sequence summary, and a helper to register CLI args.
"""

from __future__ import annotations

import os
import json
import re
import time
from dataclasses import dataclass
import math
from typing import Optional, List, Dict, Tuple


class Benchmark:
    """Minimal per-sequence benchmarking helper.

    Usage:
        bench = Benchmark(enabled=args.benchmark)
        for sequence in sequences:
            bench.reset_sequence()
            for frame in frames:
                t0 = bench.start_frame()
                ... run inference ...
                bench.end_frame(t0)
            bench.print_summary(differential_enabled=bool(diff_encoder))
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.frame_times = []  # type: list[float]
        self.first_frame = 0.0

    # Sequence lifecycle
    def reset_sequence(self) -> None:
        self.frame_times.clear()
        self.first_frame = 0.0

    # Frame lifecycle
    def start_frame(self) -> Optional[float]:
        if not self.enabled:
            return None
        return time.time()

    def end_frame(self, start_time: Optional[float]) -> None:
        if not self.enabled or start_time is None:
            return
        dt = time.time() - float(start_time)
        self.frame_times.append(dt)
        if len(self.frame_times) == 1:
            self.first_frame = dt

    # Reporting
    def has_data(self) -> bool:
        return self.enabled and bool(self.frame_times)

    def summary(self) -> dict:
        if not self.has_data():
            return {
                "total_frames": 0,
                "total_time": 0.0,
                "first_frame": 0.0,
                "avg_time_later": 0.0,
                "speedup": 1.0,
            }

        total_frames = len(self.frame_times)
        total_time = sum(self.frame_times)
        first = self.first_frame if self.frame_times else 0.0
        later = self.frame_times[1:] if total_frames > 1 else []
        avg_later = (sum(later) / len(later)) if later else first
        speedup = (first / avg_later) if avg_later > 0 else 1.0
        return {
            "total_frames": total_frames,
            "total_time": total_time,
            "first_frame": first,
            "avg_time_later": avg_later,
            "speedup": speedup,
        }

    def print_summary(self, differential_enabled: bool) -> None:
        if not self.has_data():
            return
        s = self.summary()
        print("\n" + "=" * 60)
        print("Timing Summary")
        print("=" * 60)
        print(f"Total frames: {s['total_frames']}")
        print(f"Total time: {s['total_time']:.3f}s")
        print(f"First frame time: {s['first_frame']:.3f}s")
        if s["total_frames"] > 1:
            print(f"Average time (frames 2+): {s['avg_time_later']:.3f}s")
            print(f"Speedup: {s['speedup']:.2f}x")
        print(f"Differential encoding: {'enabled' if differential_enabled else 'disabled'}")


class EncoderStatsAggregator:
    """Aggregate DifferentialVisionEncoder stats across sequences.

    Accepts per-sequence `diff_encoder.get_stats()` dicts and produces
    an overall breakdown and averages similar to existing benchmark scripts.
    """

    def __init__(self) -> None:
        self.totals: Dict[str, float] = {
            "full_encodes": 0,
            "partial_encodes": 0,
            "cache_hits": 0,
            "skipped_small": 0,
            "differential_encodes": 0,
            "total_frames": 0,
            "changed_patches_total": 0,
            "total_patches_total": 0,
        }

    def add(self, stats: Dict) -> None:
        for k in self.totals:
            self.totals[k] += float(stats.get(k, 0))

    def summary(self) -> Dict:
        t = self.totals
        avg_change_ratio = (
            (t["changed_patches_total"] / t["total_patches_total"]) if t["total_patches_total"] > 0 else 0.0
        )
        return {
            **{k: int(v) for k, v in t.items() if k not in {"changed_patches_total", "total_patches_total"}},
            "changed_patches_total": int(t["changed_patches_total"]),
            "total_patches_total": int(t["total_patches_total"]),
            "avg_change_ratio": float(avg_change_ratio),
        }

    def print_breakdown(self) -> None:
        s = self.summary()
        total_frames = max(1, s.get("total_frames", 0))
        print("\nEncoding breakdown:")
        print(f"  Full encodes: {s['full_encodes']} ({s['full_encodes']/total_frames*100:.1f}%)")
        print(f"  Partial encodes: {s['partial_encodes']} ({s['partial_encodes']/total_frames*100:.1f}%)")
        print(f"  Cache hits: {s['cache_hits']} ({s['cache_hits']/total_frames*100:.1f}%)")
        print(f"  Skipped (small): {s['skipped_small']} ({s['skipped_small']/total_frames*100:.1f}%)")
        differential = s['partial_encodes'] + s['cache_hits'] + s['skipped_small']
        print(f"  Total differential: {differential} ({differential/total_frames*100:.1f}%)")
        print(f"  Average change ratio: {s['avg_change_ratio']:.1%}")


class MultiSequenceBenchmark:
    """Aggregate timing across multiple sequences with baseline speedup.

    Mirrors summary info from the existing dataset benchmarks: throughput,
    average frames per sequence, and an estimated baseline that assumes
    full-encode latency for every frame (using average first-frame time).
    """

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.sequences: List[List[float]] = []
        self.first_frame_times: List[float] = []
        self.encoder_agg = EncoderStatsAggregator()

    def record_benchmark(self, bench: Benchmark) -> None:
        if not self.enabled or not bench.has_data():
            return
        s = bench.summary()
        self.sequences.append(list(bench.frame_times))
        self.first_frame_times.append(float(s["first_frame"]))

    def add_encoder_stats(self, stats: Dict) -> None:
        self.encoder_agg.add(stats)

    def summary(self) -> Dict:
        if not self.enabled or not self.sequences:
            return {
                "total_sequences": 0,
                "total_frames": 0,
                "total_time": 0.0,
                "avg_frames_per_sequence": 0.0,
                "avg_time_per_frame": 0.0,
                "throughput_fps": 0.0,
                "baseline_total_time": 0.0,
                "speedup_vs_baseline": 1.0,
            }

        total_frames = sum(len(seq) for seq in self.sequences)
        total_time = sum(sum(seq) for seq in self.sequences)
        total_sequences = len(self.sequences)
        avg_frames = total_frames / total_sequences if total_sequences else 0.0
        avg_time_per_frame = total_time / total_frames if total_frames else 0.0
        throughput = (total_frames / total_time) if total_time > 0 else 0.0

        avg_first = (sum(self.first_frame_times) / len(self.first_frame_times)) if self.first_frame_times else 0.0
        baseline_total = avg_first * total_frames
        speedup = (baseline_total / total_time) if total_time > 0 else 1.0

        return {
            "total_sequences": total_sequences,
            "total_frames": total_frames,
            "total_time": total_time,
            "avg_frames_per_sequence": avg_frames,
            "avg_time_per_frame": avg_time_per_frame,
            "throughput_fps": throughput,
            "baseline_total_time": baseline_total,
            "speedup_vs_baseline": speedup,
        }

    def print_overall(self, differential_enabled: bool) -> None:
        if not self.enabled or not self.sequences:
            return
        s = self.summary()
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        print(f"Total sequences: {s['total_sequences']}")
        print(f"Total frames: {s['total_frames']}")
        print(f"Total time: {s['total_time']:.2f}s")
        print(f"Avg frames/sequence: {s['avg_frames_per_sequence']:.1f}")
        print(f"Avg time/frame: {s['avg_time_per_frame']:.3f}s")
        print(f"Throughput: {s['throughput_fps']:.1f} frames/sec")
        print(f"Estimated baseline (full encode each frame): {s['baseline_total_time']:.2f}s")
        print(f"Speedup vs baseline: {s['speedup_vs_baseline']:.2f}x")
        print(f"Differential encoding: {'enabled' if differential_enabled else 'disabled'}")
        # Encoder breakdown if any was added
        self.encoder_agg.print_breakdown()


def add_benchmark_args(parser) -> None:
    """Register CLI arguments for benchmarking.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        The parser to extend.
    """
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print per-sequence timing summary.",
    )


# -------------------------
# GuiBench (trajectory I/O)
# -------------------------

ALLOWED_PYAUTOGUI_FUNCS = (
    "click",
    "write",
    "press",
    "moveTo",
    "dragTo",
    "hotkey",
    "doubleClick",
    "scroll",
    "rightClick",
    "hscroll",
)


def add_guibench_args(parser) -> None:
    """Register CLI arguments for the GUI bench runner."""
    group = parser.add_argument_group("GuiBench")
    group.add_argument(
        "--guibench",
        action="store_true",
        help="Run GUI bench on a single trajectory (agentnet_curated style)",
    )
    group.add_argument(
        "--guibench-trajectory",
        type=str,
        default=None,
        help="Path to a specific trajectory directory containing trajectory.json",
    )
    group.add_argument(
        "--guibench-root",
        type=str,
        default=None,
        help="Root directory whose subdirectories are trajectories with trajectory.json",
    )
    group.add_argument(
        "--guibench-index",
        type=int,
        default=0,
        help="Index of trajectory to pick under --guibench-root",
    )
    group.add_argument(
        "--guibench-max-steps",
        type=int,
        default=None,
        help="Optional limit on number of steps to evaluate",
    )
    group.add_argument(
        "--guibench-verbose",
        action="store_true",
        help="Print per-step predicted and expected commands",
    )
    group.add_argument(
        "--guibench-context-steps",
        type=int,
        default=0,
        help="Number of previous conversation turns to include (0 = stateless)",
    )
    group.add_argument(
        "--guibench-prompt",
        type=str,
        default=None,
        help="Custom prompt template with {action} placeholder for model-specific prompting",
    )
    group.add_argument(
        "--attention-click",
        action="store_true",
        help="Use attention-click prompt style (no parentheses except for write/press/hotkey)",
    )


def _extract_pyautogui_lines(text: str) -> List[str]:
    """Extract allowed pyautogui command lines from arbitrary text output.

    Returns a list of lines, trimmed and filtered to start with allowed functions.
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    pat = re.compile(
        r"^pyautogui\.(%s)\s*\(.*\)\s*$" % ("|".join(map(re.escape, ALLOWED_PYAUTOGUI_FUNCS))),
        re.IGNORECASE,
    )
    # In attention-click style, pointer actions may be emitted without parentheses.
    # Accept bare forms for a safe subset of functions.
    _bare_ok = {"click", "moveTo", "dragTo", "doubleClick", "rightClick"}
    pat_bare = re.compile(
        r"^pyautogui\.(%s)\s*$" % ("|".join(map(re.escape, _bare_ok))),
        re.IGNORECASE,
    )
    cmd_lines: List[str] = []
    for ln in lines:
        m = pat.match(ln)
        if m:
            # Normalize function name casing to canonical (lowercase module, func as written)
            func = m.group(1)
            # Keep original content but ensure prefix is normalized to "pyautogui.<Func>"
            rest = ln.split(".", 1)[1] if "." in ln else ln
            cmd_lines.append(f"pyautogui.{rest}")
        else:
            m2 = pat_bare.match(ln)
            if m2:
                func = m2.group(1)
                cmd_lines.append(f"pyautogui.{func}")
    return cmd_lines


def _loose_regex_from_command(cmd: str) -> re.Pattern:
    """Create a tolerant regex from a single pyautogui command string.

    - Makes whitespace flexible
    - Makes numeric literals flexible (matches any float/int in their place)
    - Keeps exact function and argument names
    """
    # Split into tokens where numeric substrings are isolated
    tokens: List[Tuple[str, bool]] = []
    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    pos = 0
    while pos < len(cmd):
        m = num_re.search(cmd, pos)
        if not m:
            tokens.append((cmd[pos:], False))
            break
        if m.start() > pos:
            tokens.append((cmd[pos:m.start()], False))
        tokens.append((cmd[m.start():m.end()], True))
        pos = m.end()

    parts: List[str] = []
    for s, is_num in tokens:
        if is_num:
            parts.append(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
        else:
            # Escape specials, then relax whitespace and space around = and ,
            esc = re.escape(s)
            esc = esc.replace(r"\ ", r"\s*")
            esc = esc.replace(r"\=", r"\s*=\s*")
            esc = esc.replace(r"\,", r"\s*,\s*")
            parts.append(esc)
    pattern = "".join(parts)
    return re.compile(r"^" + pattern + r"$", re.IGNORECASE)


def _find_trajectory_dirs(root: str) -> List[str]:
    dirs: List[str] = []
    try:
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "trajectory.json")):
                dirs.append(full)
    except FileNotFoundError:
        pass
    return dirs


@dataclass
class GuiBenchResult:
    trajectory_dir: str
    total_steps: int
    eval_steps: int
    expected_actions: int
    matched_actions: int
    accuracy: float


class GuiBench:
    """GUI action prediction benchmark over a single trajectory.

    For each step in `trajectory.json` under a trajectory directory, feed the
    screenshot image and the natural-language "action" to the VLM, asking it to
    output only `pyautogui.*` commands. The predicted commands are compared to
    any ground-truth commands present in the JSON (when available) using a
    tolerant regex match.
    """

    def __init__(self, trajectory_dir: str, max_steps: Optional[int] = None, verbose: bool = False) -> None:
        self.trajectory_dir = os.path.expanduser(trajectory_dir)
        self.max_steps = max_steps
        self.verbose = verbose

    @staticmethod
    def pick_trajectory(root: Optional[str], index: int = 0) -> Optional[str]:
        if not root:
            return None
        root = os.path.expanduser(root)
        dirs = _find_trajectory_dirs(root)
        if not dirs:
            return None
        index = max(0, min(index, len(dirs) - 1))
        return dirs[index]

    def _load_steps(self) -> List[Dict]:
        path = os.path.join(self.trajectory_dir, "trajectory.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        steps = data.get("steps", [])
        if self.max_steps is not None:
            steps = steps[: self.max_steps]
        return steps

    @staticmethod
    def _resolve_image_path(base_dir: str, step: Dict) -> Optional[str]:
        key_candidates = ["image_filename", "image", "image_path", "screenshot"]
        image_name = None
        for k in key_candidates:
            if k in step and step[k]:
                image_name = step[k]
                break
        if not image_name:
            return None
        full = os.path.join(base_dir, image_name)
        if os.path.isfile(full):
            return full
        # Try basename lookup if nested
        base = os.path.basename(image_name)
        full2 = os.path.join(base_dir, base)
        return full2 if os.path.isfile(full2) else None

    @staticmethod
    def _extract_expected_commands(step: Dict) -> List[str]:
        # Try several common fields
        candidates: List[str] = []
        for key in ("pyautogui", "code", "commands", "script", "gt_code", "expected"):
            val = step.get(key)
            if isinstance(val, list):
                candidates.extend([str(v) for v in val])
            elif isinstance(val, str):
                candidates.append(val)
        if not candidates:
            return []
        # Extract allowed command lines from the concatenated text
        joined = "\n".join(candidates)
        return _extract_pyautogui_lines(joined)

    @staticmethod
    def _get_action_text(step: Dict) -> str:
        # The NL instruction for what to do
        return str(step.get("action", ""))

    def run(self, *, tokenizer, model, image_processor, device, conv_mode: str = "qwen_2", temperature: float = 0.2, top_p: Optional[float] = None, num_beams: int = 1, custom_prompt: Optional[str] = None, context_steps: int = 0, attention_click_prompt: bool = False) -> GuiBenchResult:
        # Local imports to avoid heavy deps at module import time
        import torch
        from PIL import Image
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        steps = self._load_steps()
        if not steps:
            raise ValueError(f"No steps found in {self.trajectory_dir}/trajectory.json")

        # Maintain prior turns as plain text (without <image> tokens)
        history: List[Tuple[str, str]] = []

        # Dtype alignment
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float16 if device.type != "cpu" else torch.float32

        total_expected = 0
        matched = 0

        for idx, step in enumerate(steps, start=1):
            img_path = self._resolve_image_path(self.trajectory_dir, step)
            if not img_path or not os.path.isfile(img_path):
                if self.verbose:
                    print(f"[Guibench] Skip step {idx}: image not found: {img_path}")
                continue

            # Build user message: image token prefix + action instruction
            action_text = self._get_action_text(step)
            if custom_prompt:
                # Allow custom prompt template with {action} placeholder
                user_plain = custom_prompt.format(action=action_text)
            else:
                if attention_click_prompt:
                    # Attention-click style: no parentheses for pointer actions; keep for write/press/hotkey
                    instr = (
                        "Look at the image and predict the exact screen coordinates for this action.\n"
                        "Output ONLY pyautogui command name\n"
                        "Command formats:\n"
                        "pyautogui.click\n"
                        "pyautogui.rightClick\n"
                        "pyautogui.doubleClick\n"
                        "pyautogui.moveTo\n"
                        "pyautogui.dragTo\n"
                        "pyautogui.scroll <amount>\n"
                        "pyautogui.hscroll <amount>\n"
                        "pyautogui.write(message='<text>')\n"
                        "pyautogui.press('<key_name>')\n"
                        "pyautogui.hotkey('<key1>', '<key2>')\n\n"
                        "Action:"
                    )
                else:
                    # Default style: include parentheses for all examples
                    instr = (
                        "Look at the image and predict the exact screen coordinates for this action.\n"
                        "Output ONLY valid pyautogui commands. Use normalized coordinates (0.0 to 1.0) based on what you see.\n"
                        "Command formats:\n"
                        "pyautogui.click(x=<screen_x>, y=<screen_y>)\n"
                        "pyautogui.rightClick(x=<screen_x>, y=<screen_y>)\n"
                        "pyautogui.doubleClick(x=<screen_x>, y=<screen_y>)\n"
                        "pyautogui.moveTo(x=<screen_x>, y=<screen_y>)\n"
                        "pyautogui.dragTo(x=<screen_x>, y=<screen_y>)\n"
                        "pyautogui.write(message='<text>')\n"
                        "pyautogui.press('<key_name>')\n"
                        "pyautogui.hotkey('<key1>', '<key2>')\n"
                        "pyautogui.scroll(<amount>)\n"
                        "pyautogui.hscroll(<amount>)\n\n"
                        "Action:"
                    )
                user_plain = f"{instr}\n{action_text}"
            # Rebuild conversation fresh each step so only current turn carries the <image> token
            conv = conv_templates[conv_mode].copy()
            # Include only the last `context_steps` prior turns (0 => stateless)
            if context_steps > 0:
                for u, a in history[-int(context_steps):]:
                    conv.append_message(conv.roles[0], u)
                    conv.append_message(conv.roles[1], a)
            user_with_image = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_plain
                if getattr(model.config, "mm_use_im_start_end", False)
                else DEFAULT_IMAGE_TOKEN + "\n" + user_plain
            )
            conv.append_message(conv.roles[0], user_with_image)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Prepare inputs
            pil_img = Image.open(img_path).convert("RGB")
            processed = process_images([pil_img], image_processor, model.config)
            image_tensor = processed[0] if isinstance(processed, (list, tuple)) else processed
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device=device, dtype=model_dtype)

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[pil_img.size],
                    do_sample=bool(temperature) and temperature > 0,
                    temperature=float(temperature),
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=64,
                    use_cache=True,
                )

            # Decode the first-pass text early so we can use it for heatmap query aggregation
            pred_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Also extract expected commands early to enable GT marker on the heatmap
            exp_cmds = self._extract_expected_commands(step)

            # Save attention heatmap overlay whenever attention-click prompt style is used
            # Keep a handle to the last computed heatmap for optional second-pass picking
            last_heatmap = None
            if attention_click_prompt:
                # Default output location and params
                default_target = "heatmaps"
                default_layer = -1
                default_alpha = 0.45
                default_mode = "smooth"

                expects_multi = len(steps) > 1
                heatmap_path = _resolve_heatmap_path(default_target, img_path, expects_multi)
                if heatmap_path:
                    try:
                        # Extract ground truth coordinates from expected commands
                        gt_coords = None
                        for exp_cmd in exp_cmds:
                            coords = _parse_coordinates_from_command(exp_cmd)
                            if coords is not None:
                                gt_coords = coords
                                break  # Use first coordinate-based command

                        # Rebuild a conversation INCLUDING the generated assistant text,
                        # so the last token belongs to pred_text (not the instruction).
                        attn_conv = conv_templates[conv_mode].copy()
                        if context_steps > 0:
                            for u, a in history[-int(context_steps):]:
                                attn_conv.append_message(attn_conv.roles[0], u)
                                attn_conv.append_message(attn_conv.roles[1], a)
                        attn_user = (
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_plain
                            if getattr(model.config, "mm_use_im_start_end", False)
                            else DEFAULT_IMAGE_TOKEN + "\n" + user_plain
                        )
                        attn_conv.append_message(attn_conv.roles[0], attn_user)
                        # IMPORTANT: include model's generated text as assistant message
                        attn_conv.append_message(attn_conv.roles[1], pred_text or "")
                        attn_prompt = attn_conv.get_prompt()
                        attn_ids = tokenizer_image_token(
                            attn_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        ).unsqueeze(0).to(device)

                        # Fused multi-layer vector rollout using user-last query strategy
                        heatmap = _extract_pointer_heatmap(
                            model,
                            attn_ids,
                            image_tensor,
                            pil_img.size,
                            user_input_len=input_ids.shape[1],
                            last_k_layers=6,
                            query_strategy="action",
                            query_last_k=8,
                            sparse_norm="sparsemax",
                            head_weighting=True,
                            tokenizer=tokenizer,
                            action_text=action_text,
                        )
                        last_heatmap = heatmap
                        if self.verbose and heatmap is not None:
                            try:
                                hmin = float(heatmap.min().item())
                                hmax = float(heatmap.max().item())
                                hmean = float(heatmap.mean().item())
                                print(f"[heatmap] stats: shape={list(heatmap.shape)}, min={hmin:.4f}, max={hmax:.4f}, mean={hmean:.4f}")
                            except Exception:
                                pass
                        # Optional quick A/B metric: pixel distance from peak to ground truth
                        if heatmap is not None and gt_coords is not None:
                            import torch as _torch
                            yx = _torch.stack(_torch.meshgrid(
                                _torch.arange(heatmap.shape[0]),
                                _torch.arange(heatmap.shape[1]),
                                indexing="ij"
                            ), dim=-1).reshape(-1, 2).to(_torch.float32)
                            idx = heatmap.view(-1).argmax().item()
                            peak = yx[idx]
                            peak_px = (peak[1] * pil_img.width / heatmap.shape[1],
                                       peak[0] * pil_img.height / heatmap.shape[0])
                            gx, gy = gt_coords
                            if 0 <= gx <= 1 and 0 <= gy <= 1:
                                gt_px = (gx * pil_img.width, gy * pil_img.height)
                            else:
                                gt_px = (gx, gy)
                            dist = ((peak_px[0] - gt_px[0]) ** 2 + (peak_px[1] - gt_px[1]) ** 2) ** 0.5
                            print(f"[pointer] pixel_distance={dist:.1f}")
                        # Build heatmap overlay, drawing candidate boxes as well
                        overlay = None
                        if heatmap is not None:
                            try:
                                # Lazily import targeter only when needed
                                from attention_click.click_targeter import ClickTargeter
                                from PIL import ImageDraw
                                # Generate top-K candidates for visualization
                                K = 4
                                targeter = ClickTargeter(max_candidates=K)
                                candidates = targeter.generate_candidates(
                                    heatmap,
                                    pil_img.size,
                                    adaptive_threshold=True,
                                    interpolation_mode=("bilinear" if str(default_mode) != "tile" else "nearest"),
                                )
                                # Create overlay with heatmap and boxes
                                overlay = targeter.visualize_candidates(
                                    pil_img,
                                    candidates=candidates,
                                    heatmap=heatmap,
                                    alpha=float(default_alpha),
                                    interpolation_mode=("bilinear" if str(default_mode) != "tile" else "nearest"),
                                )
                                # Draw ground-truth marker on top if available
                                if gt_coords is not None:
                                    x, y = gt_coords
                                    if 0 <= x <= 1 and 0 <= y <= 1:
                                        px = int(x * pil_img.width)
                                        py = int(y * pil_img.height)
                                    else:
                                        px = int(x)
                                        py = int(y)
                                    draw = ImageDraw.Draw(overlay)
                                    radius = 12
                                    # Outer white ring
                                    draw.ellipse(
                                        [(px - radius - 2, py - radius - 2), (px + radius + 2, py + radius + 2)],
                                        outline="white",
                                        width=3,
                                    )
                                    # Inner cyan disc
                                    draw.ellipse(
                                        [(px - radius, py - radius), (px + radius, py + radius)],
                                        fill="cyan",
                                        outline="black",
                                        width=2,
                                    )

                                # Additionally: run a second pass using the default (parentheses) style
                                # to obtain the model's predicted coordinates, and mark them on the heatmap.
                                try:
                                    # Build default-style instruction (with parentheses examples)
                                    default_instr = (
                                        "You are a helpful multi-modal assistant controlling a GUI via pyautogui commands only.\n"
                                        "Output ONLY valid pyautogui commands. Use normalized coordinates (0.0 to 1.0).\n"
                                        "Command formats:\n"
                                        "pyautogui.click(x=<screen_x>, y=<screen_y>)\n"
                                        "pyautogui.rightClick(x=<screen_x>, y=<screen_y>)\n"
                                        "pyautogui.doubleClick(x=<screen_x>, y=<screen_y>)\n"
                                        "pyautogui.moveTo(x=<screen_x>, y=<screen_y>)\n"
                                        "pyautogui.dragTo(x=<screen_x>, y=<screen_y>)\n"
                                        "pyautogui.write(message='<text>')\n"
                                        "pyautogui.press('<key_name>')\n"
                                        "pyautogui.hotkey('<key1>', '<key2>')\n"
                                        "pyautogui.scroll(<amount>)\n"
                                        "pyautogui.hscroll(<amount>)\n\n"
                                        "Action:"
                                    )
                                    user_plain_default = f"{default_instr}\n{action_text}"

                                    # Rebuild conversation for the default-style pass (stateless or with requested context)
                                    norm_conv = conv_templates[conv_mode].copy()
                                    if context_steps > 0:
                                        for u, a in history[-int(context_steps):]:
                                            norm_conv.append_message(norm_conv.roles[0], u)
                                            norm_conv.append_message(norm_conv.roles[1], a)
                                    norm_user = (
                                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_plain_default
                                        if getattr(model.config, "mm_use_im_start_end", False)
                                        else DEFAULT_IMAGE_TOKEN + "\n" + user_plain_default
                                    )
                                    norm_conv.append_message(norm_conv.roles[0], norm_user)
                                    norm_conv.append_message(norm_conv.roles[1], None)
                                    norm_prompt = norm_conv.get_prompt()

                                    norm_ids = tokenizer_image_token(
                                        norm_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                                    ).unsqueeze(0).to(device)

                                    with torch.inference_mode():
                                        norm_out = model.generate(
                                            norm_ids,
                                            images=image_tensor,
                                            image_sizes=[pil_img.size],
                                            do_sample=bool(temperature) and temperature > 0,
                                            temperature=float(temperature),
                                            top_p=top_p,
                                            num_beams=num_beams,
                                            max_new_tokens=64,
                                            use_cache=True,
                                        )

                                    norm_text = tokenizer.batch_decode(norm_out, skip_special_tokens=True)[0].strip()
                                    norm_cmds = _extract_pyautogui_lines(norm_text)
                                    # Pick the first coordinate-based command
                                    pred_coords = None
                                    for cmd in norm_cmds:
                                        pred_coords = _parse_coordinates_from_command(cmd)
                                        if pred_coords is not None:
                                            break

                                    if pred_coords is not None:
                                        mx, my = pred_coords
                                        # Normalize to pixel space if input is 0..1
                                        if 0.0 <= mx <= 1.0 and 0.0 <= my <= 1.0:
                                            mx_px = int(round(mx * pil_img.width))
                                            my_px = int(round(my * pil_img.height))
                                        else:
                                            mx_px = int(round(mx))
                                            my_px = int(round(my))

                                        # Draw predicted marker in a distinct color (magenta with white outline)
                                        draw = ImageDraw.Draw(overlay)
                                        pr = 10
                                        draw.ellipse(
                                            [(mx_px - pr - 2, my_px - pr - 2), (mx_px + pr + 2, my_px + pr + 2)],
                                            outline="white",
                                            width=3,
                                        )
                                        draw.ellipse(
                                            [(mx_px - pr, my_px - pr), (mx_px + pr, my_px + pr)],
                                            fill="magenta",
                                            outline="black",
                                            width=2,
                                        )
                                except Exception as _exc_pred_marker:
                                    if self.verbose:
                                        print(f"[heatmap] predicted-dot overlay skipped: {_exc_pred_marker}")
                            except Exception:
                                # Fallback to plain heatmap overlay if candidate drawing fails
                                overlay = _render_heatmap_overlay(
                                    pil_img,
                                    heatmap,
                                    alpha=float(default_alpha),
                                    mode=str(default_mode),
                                    ground_truth_coords=gt_coords,
                                )
                        if overlay is not None:
                            overlay.save(heatmap_path)
                            if self.verbose:
                                print(f"[heatmap] saved overlay: {heatmap_path}")
                        else:
                            print(f"[heatmap] Skipping {os.path.basename(img_path)} (no vision tokens or attention available).")
                    except Exception as exc:
                        print(f"[heatmap] Failed to generate heatmap for {img_path}: {exc}")



            if self.verbose:
                print(f"[DEBUG] output_ids shape: {output_ids.shape}")
                print(f"[DEBUG] input_ids shape: {input_ids.shape}")
                # output_ids should contain input+generated; compute actual generated count
                gen_count = output_ids.shape[1] - input_ids.shape[1] if output_ids.shape[1] >= input_ids.shape[1] else output_ids.shape[1]
                print(f"[DEBUG] Generated tokens: {gen_count}")

            # Save turn into plain-text history for the next step
            history.append((user_plain, pred_text))

            pred_cmds = _extract_pyautogui_lines(pred_text)

            if self.verbose:
                print(f"\n[Step {idx}] {os.path.basename(img_path)}")
                print("Action:", action_text)
                if exp_cmds:
                    print("Expected:")
                    for ln in exp_cmds:
                        print("  ", ln)
                print("Predicted:")
                # Debug: show raw model output if no commands extracted
                if not pred_cmds and pred_text:
                    print(f"  [DEBUG] Raw model output (no pyautogui commands found):")
                    print(f"  [DEBUG] {repr(pred_text)}")
                elif not pred_text:
                    print(f"  [DEBUG] Model returned empty output!")
                for ln in pred_cmds:
                    print("  ", ln)

            # Optional second forward pass: inject top-K candidate boxes and ask model to pick one
            # This runs only in attention-click mode and only if a heatmap was computed
            final_pick_cmds: List[str] = []
            if attention_click_prompt and last_heatmap is not None:
                try:
                    # Lazily import click targeter to keep dependencies local
                    from attention_click.click_targeter import ClickTargeter

                    K = 4
                    targeter = ClickTargeter(max_candidates=K)
                    candidates = targeter.generate_candidates(
                        last_heatmap, pil_img.size, adaptive_threshold=True
                    )

                    if candidates:
                        # Limit to top-K
                        candidates = candidates[:K]
                        if self.verbose:
                            print(f"[second-pass] candidates: requested={K}, generated={len(candidates)}")
                            for i, c in enumerate(candidates, 1):
                                x1, y1, x2, y2 = c.bbox
                                cx, cy = c.center
                                print(f"  {i}. bbox=({x1},{y1},{x2},{y2}) center=({cx},{cy}) score={c.attention_score:.3f} area={c.area}")

                        # Build compact numbered list for the prompt
                        def _format_boxes_for_prompt(cands) -> str:
                            lines = []
                            for i, c in enumerate(cands, 1):
                                x1, y1, x2, y2 = c.bbox
                                cx, cy = c.center
                                lines.append(
                                    f"Box {i}: ({x1},{y1})-({x2},{y2}), center=({cx},{cy})"
                                )
                            return "\n".join(lines)

                        boxes_desc = _format_boxes_for_prompt(candidates)
                        if self.verbose:
                            print("[second-pass] boxes description:\n" + boxes_desc)

                        # Create an instruction that asks the model to choose one box and output a single click
                        # Keep style consistent with attention-click prompt (only pyautogui commands)
                        pick_instr = (
                            "You now have 4 candidate regions (Box 1..4).\n"
                            "Pick the single best box and output exactly one line:\n"
                            "pyautogui.click(x=<int>, y=<int>)\n"
                            "No other text."
                        )

                        # Rebuild a conversation for the picking step including prior context
                        pick_conv = conv_templates[conv_mode].copy()
                        if context_steps > 0:
                            for u, a in history[-int(context_steps):]:
                                pick_conv.append_message(pick_conv.roles[0], u)
                                pick_conv.append_message(pick_conv.roles[1], a)

                        pick_user = (
                            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
                            if getattr(model.config, "mm_use_im_start_end", False)
                            else DEFAULT_IMAGE_TOKEN + "\n"
                        )
                        # Include the original action, the model's first answer, and the compact boxes list
                        pick_user += (
                            f"Action: {action_text}\n"
                            f"First pass: {pred_text}\n"
                            f"Candidates:\n{boxes_desc}\n\n{pick_instr}"
                        )

                        pick_conv.append_message(pick_conv.roles[0], pick_user)
                        pick_conv.append_message(pick_conv.roles[1], None)
                        pick_prompt = pick_conv.get_prompt()

                        pick_ids = tokenizer_image_token(
                            pick_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        ).unsqueeze(0).to(device)

                        with torch.inference_mode():
                            pick_out = model.generate(
                                pick_ids,
                                images=image_tensor,
                                image_sizes=[pil_img.size],
                                do_sample=False,
                                temperature=0.0,
                                max_new_tokens=32,
                                use_cache=True,
                            )
                        pick_text = tokenizer.batch_decode(pick_out, skip_special_tokens=True)[0].strip()
                        if self.verbose:
                            try:
                                gen2 = pick_out.shape[1] - pick_ids.shape[1] if pick_out.shape[1] >= pick_ids.shape[1] else pick_out.shape[1]
                                print(f"[second-pass] output_ids shape: {pick_out.shape}, generated tokens: {gen2}")
                            except Exception:
                                pass
                            print("[second-pass] raw selection output:")
                            print("  " + repr(pick_text))

                        # Extract final click command from the second pass
                        final_pick_cmds = _extract_pyautogui_lines(pick_text)

                        if self.verbose:
                            print("Refined (second pass):")
                            if not final_pick_cmds and pick_text:
                                print("  [DEBUG] Raw refined output (no pyautogui found):")
                                print(f"  [DEBUG] {repr(pick_text)}")
                            for ln in final_pick_cmds:
                                print("  ", ln)
                    else:
                        if self.verbose:
                            print("[second-pass] No candidates generated from heatmap; skipping selection")
                except Exception as exc:
                    if self.verbose:
                        print(f"[second-pass] Skipped due to error: {exc}")

            # Score: count of expected commands matched by prediction (first-pass)
            step_hits = 0
            if exp_cmds:
                total_expected += len(exp_cmds)
                # Pre-compile tolerant regex for each expected command
                patterns = [_loose_regex_from_command(c) for c in exp_cmds]
                for pat in patterns:
                    if any(pat.match(p) for p in pred_cmds):
                        step_hits += 1
                matched += step_hits

            # If a refined single click was produced, print a final convenience line
            if final_pick_cmds:
                try:
                    # Print only the first refined click as the final action suggestion
                    print("Final:")
                    print(f"  {final_pick_cmds[0]}")
                    if self.verbose:
                        print("[second-pass] Using first refined command as final suggestion")
                except Exception:
                    pass

        accuracy = (matched / total_expected) if total_expected > 0 else 0.0
        return GuiBenchResult(
            trajectory_dir=self.trajectory_dir,
            total_steps=len(steps),
            eval_steps=len(steps),
            expected_actions=total_expected,
            matched_actions=matched,
            accuracy=accuracy,
        )


# -------------------------
# Heatmap helpers (mirrors predict.py)
# -------------------------

def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cpu")


def _infer_patch_grid(model, num_tokens: int):
    tower = model.get_model().get_vision_tower()
    if isinstance(tower, list):
        tower = tower[0]
    side = getattr(tower, "num_patches_per_side", None)
    if isinstance(side, int) and side * side == num_tokens:
        return side, side
    root = int(math.sqrt(num_tokens))
    for height in range(root, 0, -1):
        if num_tokens % height == 0:
            return height, num_tokens // height
    return num_tokens, 1


def _extract_image_token_counts(model, image_tensor, image_size):
    import torch
    with torch.inference_mode():
        features = model.encode_images(image_tensor)
    if isinstance(features, tuple):
        features = features[0]
    if isinstance(features, (list, tuple)):
        return [feat.shape[0] for feat in features]
    if hasattr(features, "ndim"):
        if features.ndim == 3:  # [B, N, D]
            return [features[i].shape[0] for i in range(features.shape[0])]
        if features.ndim == 2:  # [N, D]
            return [features.shape[0]]
    raise ValueError("Unsupported vision feature shape for heatmap extraction.")


def _build_image_token_mask(sequence_tokens, image_token_counts, total_length, device):
    import torch
    if not image_token_counts:
        return None
    mask = torch.zeros(total_length, dtype=torch.bool, device=device)
    cursor = 0
    image_idx = 0
    from llava.constants import IMAGE_TOKEN_INDEX
    for token in sequence_tokens:
        if cursor >= total_length:
            break
        if token == IMAGE_TOKEN_INDEX:
            if image_idx >= len(image_token_counts):
                break
            count = int(image_token_counts[image_idx])
            end = min(cursor + count, total_length)
            mask[cursor:end] = True
            cursor = end
            image_idx += 1
        else:
            cursor += 1
    return mask if mask.any() else None


def _extract_attention_heatmap(
    model,
    prompt_ids,
    image_tensor,
    image_size,
    layer_index: int = -1,
    query_last_n: Optional[int] = None,
):
    import torch
    device = _model_device(model)

    # Force eager attention to get full attention maps
    original_attn_impl = None
    attn_setter = getattr(model, "set_attn_implementation", None)
    used_setter = False
    if attn_setter is not None:
        try:
            current_impl = getattr(model.config, "_attn_implementation", None)
            if current_impl != "eager":
                original_attn_impl = current_impl
                attn_setter("eager")
                used_setter = True
        except Exception:
            pass
    elif hasattr(model.config, "_attn_implementation"):
        current_impl = getattr(model.config, "_attn_implementation")
        if current_impl != "eager":
            original_attn_impl = current_impl
            model.config._attn_implementation = "eager"

    try:
        with torch.inference_mode():
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = model.prepare_inputs_labels_for_multimodal(
                prompt_ids,
                None,
                None,
                None,
                None,
                image_tensor,
                image_sizes=[image_size],
            )

        inputs_embeds = inputs_embeds.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            valid_len = int(attention_mask[0].sum().item())
        else:
            valid_len = inputs_embeds.shape[1]

        outputs = model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            return_dict=True,
        )
    finally:
        if original_attn_impl is not None:
            try:
                if used_setter and attn_setter is not None:
                    attn_setter(original_attn_impl)
                elif hasattr(model.config, "_attn_implementation"):
                    model.config._attn_implementation = original_attn_impl
            except Exception:
                pass

    attentions = outputs.attentions
    if not attentions:
        return None

    layer_count = len(attentions)
    layer_idx = layer_index if layer_index >= 0 else layer_count + layer_index
    layer_idx = max(0, min(layer_idx, layer_count - 1))

    layer_attn = attentions[layer_idx][0]
    layer_attn = layer_attn[:, :valid_len, :valid_len]
    layer_attn = torch.nan_to_num(layer_attn, nan=0.0, posinf=0.0, neginf=0.0)

    # Aggregate attention from the last N tokens (default: last token)
    qlen = int(query_last_n) if (query_last_n is not None and int(query_last_n) > 0) else 1
    qlen = min(qlen, layer_attn.shape[1])
    q_start = layer_attn.shape[1] - qlen
    attn_vec = layer_attn[:, q_start:, :]  # [heads, qlen, seq_len]
    attn_mean = attn_vec.mean(dim=(0, 1))  # average over heads and selected query tokens -> [seq_len]
    attn_mean = torch.nan_to_num(attn_mean, nan=0.0, posinf=0.0, neginf=0.0)

    sequence_tokens = prompt_ids[0].tolist()
    image_token_counts = _extract_image_token_counts(model, image_tensor, image_size)
    mask = _build_image_token_mask(sequence_tokens, image_token_counts, layer_attn.shape[-1], attn_mean.device)
    if mask is None:
        return None

    image_attn = attn_mean[mask]
    grid_h, grid_w = _infer_patch_grid(model, image_attn.shape[0])
    if grid_h * grid_w != image_attn.shape[0]:
        return None
    return image_attn.reshape(grid_h, grid_w).to(torch.float32)


def _parse_coordinates_from_command(cmd: str) -> Optional[Tuple[float, float]]:
    """Extract x, y coordinates from a pyautogui command.
    
    Supports formats like:
    - pyautogui.click(x=0.5, y=0.3)
    - pyautogui.moveTo(x=100, y=200)
    - pyautogui.click  # attention-click format (no coords)
    
    Returns coordinates as (x, y) or None if not a coordinate-based command.
    """
    if not cmd or "pyautogui" not in cmd.lower():
        return None
    
    # Check if this is a coordinate-based command
    coordinate_funcs = {"click", "moveto", "dragto", "doubleclick", "rightclick"}
    func_name = None
    for func in coordinate_funcs:
        if f"pyautogui.{func}" in cmd.lower():
            func_name = func
            break
    
    if not func_name:
        return None
    
    # Try to extract x= and y= parameters
    import re
    x_match = re.search(r'x\s*=\s*([-+]?\d*\.?\d+)', cmd, re.IGNORECASE)
    y_match = re.search(r'y\s*=\s*([-+]?\d*\.?\d+)', cmd, re.IGNORECASE)
    
    if x_match and y_match:
        try:
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            return (x, y)
        except ValueError:
            pass
    
    return None


def _render_heatmap_overlay(image, heatmap, alpha: float, mode: str = "smooth", ground_truth_coords: Optional[Tuple[float, float]] = None):
    if heatmap is None:
        return None
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    
    alpha = max(0.0, min(float(alpha), 1.0))
    heat = heatmap
    finite_mask = torch.isfinite(heat)
    if finite_mask.any():
        minv = heat[finite_mask].min()
        heat = heat - minv
        maxv = heat[finite_mask].max()
        if maxv > 0:
            heat = heat / maxv
    else:
        heat = torch.zeros_like(heat)
    heat = heat.unsqueeze(0).unsqueeze(0)
    interp_mode = "bilinear" if mode != "tile" else "nearest"
    alignable = {"linear", "bilinear", "bicubic", "trilinear"}
    interp_kwargs = {}
    if interp_mode in alignable:
        interp_kwargs["align_corners"] = False
    heat = torch.nn.functional.interpolate(
        heat,
        size=(image.height, image.width),
        mode=interp_mode,
        **interp_kwargs,
    ).squeeze(0).squeeze(0).detach().cpu().numpy()
    heat = np.nan_to_num(heat, nan=0.0, posinf=1.0, neginf=0.0)

    base = np.array(image.convert("RGB")).astype(np.float32)
    heat = np.clip(heat, 0.0, 1.0)[..., None]
    alpha_map = heat * alpha
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    overlay = base * (1.0 - alpha_map) + color * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    result = Image.fromarray(overlay)
    
    # Draw ground truth coordinate marker if provided
    if ground_truth_coords is not None:
        x, y = ground_truth_coords
        # Convert normalized coordinates (0-1) to pixel coordinates if needed
        if 0 <= x <= 1 and 0 <= y <= 1:
            pixel_x = int(x * image.width)
            pixel_y = int(y * image.height)
        else:
            pixel_x = int(x)
            pixel_y = int(y)
        
        # Draw a cyan circle with white outline to mark ground truth
        draw = ImageDraw.Draw(result)
        radius = 12
        # White outer circle for visibility
        draw.ellipse(
            [(pixel_x - radius - 2, pixel_y - radius - 2),
             (pixel_x + radius + 2, pixel_y + radius + 2)],
            outline="white",
            width=3
        )
        # Cyan inner circle
        draw.ellipse(
            [(pixel_x - radius, pixel_y - radius),
             (pixel_x + radius, pixel_y + radius)],
            fill="cyan",
            outline="black",
            width=2
        )
    
    return result


def _resolve_heatmap_path(target, image_path, expects_multiple: bool):
    if target is None:
        return None
    is_directory = (
        expects_multiple
        or target.endswith(os.sep)
        or os.path.isdir(target)
        or os.path.splitext(target)[1] == ""
    )
    if is_directory:
        os.makedirs(target, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(target, f"{stem}_heatmap.png")
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return target

# -------------------------
# Pointer rollout helpers
# -------------------------

def _sparsemax(logits, dim=-1, eps=1e-12):
    import torch
    z = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(z, dim=dim, descending=True).values
    range_ = torch.arange(1, zs.size(dim) + 1, device=zs.device, dtype=zs.dtype).view(
        *([1] * (zs.dim() - 1)), -1
    )
    cumsum = zs.cumsum(dim=dim)
    cond = zs > (cumsum - 1) / range_
    k = cond.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (cumsum.gather(dim, k - 1) - 1) / k
    return torch.clamp(z - tau, min=0.0)


def _select_query_indices(strategy, attn_len, user_len, gen_len, last_k=8):
    """
    Returns a 1D LongTensor of query token indices in [0, attn_len).
    user_len: tokens up to and including the end of the USER message.
    gen_len: number of assistant-generated tokens appended.
    """
    import torch
    if strategy == "assistant_last":
        qlen = max(1, min(int(gen_len), int(last_k)))
        return torch.arange(attn_len - qlen, attn_len, device="cpu")
    elif strategy == "user_last":
        qlen = min(int(last_k), int(user_len))
        return torch.arange(int(user_len) - qlen, int(user_len), device="cpu")
    else:
        qlen = min(int(last_k), int(user_len))
        return torch.arange(int(user_len) - qlen, int(user_len), device="cpu")


def _tokenize_plain(tokenizer, text: str):
    # Tokenize without adding BOS/EOS
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return ids


def _find_subsequence_indices(haystack_ids, needle_ids):
    """Return all start indices where needle occurs in haystack."""
    matches = []
    if not needle_ids or len(needle_ids) > len(haystack_ids):
        return matches
    for i in range(0, len(haystack_ids) - len(needle_ids) + 1):
        if haystack_ids[i : i + len(needle_ids)] == needle_ids:
            matches.append(i)
    return matches


def _select_action_token_indices(attn_ids, user_input_len, tokenizer, action_text: str):
    """
    Find token indices (in attn_ids) that correspond to `action_text` within the user turn
    (positions < user_input_len). Returns a 1D LongTensor (may be empty).
    """
    import torch
    user_ids = attn_ids[0, : int(user_input_len)].tolist()
    cand_texts = [
        action_text,
        action_text.strip().strip('"').strip("'"),
        f'"{action_text}"',
        f"'{action_text}'",
    ]
    hits = []
    for t in cand_texts:
        needle = _tokenize_plain(tokenizer, t)
        if not needle:
            continue
        starts = _find_subsequence_indices(user_ids, needle)
        for s in starts:
            hits.extend(range(s, s + len(needle)))
        if hits:
            break
    if not hits:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(set(hits)), dtype=torch.long)


def _extract_pointer_heatmap(
    model,
    attn_ids,              # ids for [user message (+ image)] + assistant text (optional)
    image_tensor,
    image_size,            # (W, H)
    *,
    user_input_len,        # length of ids up to end of user turn (no assistant)
    last_k_layers=6,
    query_strategy="user_last",    # "user_last" | "assistant_last" | "action"
    query_last_k=8,
    sparse_norm="sparsemax",       # "sparsemax" | None
    head_weighting=True,
    tokenizer=None,
    action_text=None,
):
    """
    Vector rollout over the last K layers, returns [Hgrid, Wgrid] float heatmap on image tokens.
    """
    import torch
    device = _model_device(model)

    # Force eager attention to expose maps
    original_attn_impl, attn_setter, used_setter = None, getattr(model, "set_attn_implementation", None), False
    try:
        current_impl = getattr(model.config, "_attn_implementation", None)
        if current_impl != "eager":
            original_attn_impl = current_impl
            if attn_setter:
                attn_setter("eager"); used_setter = True
            else:
                model.config._attn_implementation = "eager"
    except Exception:
        pass

    try:
        with torch.inference_mode():
            (
                _,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = model.prepare_inputs_labels_for_multimodal(
                attn_ids, None, None, None, None, image_tensor, image_sizes=[image_size]
            )
        inputs_embeds = inputs_embeds.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            valid_len = int(attention_mask[0].sum().item())
        else:
            valid_len = inputs_embeds.shape[1]

        # Forward with attentions
        outputs = model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            return_dict=True,
        )
        attn_stack = outputs.attentions  # list length L: [B=1, H, T, T]
        if not attn_stack:
            return None

        L = len(attn_stack)
        K = max(1, min(int(last_k_layers), L))
        layer_ids = list(range(L - K, L))

        # Build image mask over sequence positions at this step
        sequence_tokens = attn_ids[0].tolist()
        image_token_counts = _extract_image_token_counts(model, image_tensor, image_size)
        img_mask = _build_image_token_mask(sequence_tokens, image_token_counts, valid_len, device=device)
        if img_mask is None or not img_mask.any():
            return None

        # Query indices
        gen_len = int(attn_ids.shape[1] - int(user_input_len))
        if query_strategy == "action" and tokenizer is not None and action_text:
            q_idx = _select_action_token_indices(attn_ids, int(user_input_len), tokenizer, action_text).to(device)
            if q_idx.numel() == 0:
                q_idx = _select_query_indices("user_last", valid_len, int(user_input_len), gen_len, last_k=int(query_last_k)).to(device)
        elif query_strategy == "assistant_last":
            q_idx = _select_query_indices("assistant_last", valid_len, int(user_input_len), gen_len, last_k=int(query_last_k)).to(device)
        else:
            q_idx = _select_query_indices("user_last", valid_len, int(user_input_len), gen_len, last_k=int(query_last_k)).to(device)

        # Init rollout vector r0: average one-hot over chosen queries
        r = torch.zeros(valid_len, device=device, dtype=inputs_embeds.dtype)
        denom = max(1, int(q_idx.numel()))
        if denom > 0:
            r[q_idx] = 1.0 / float(denom)

        # Roll through layers
        for lid in layer_ids:
            A = attn_stack[lid][0][:, :valid_len, :valid_len]  # [H, T, T]
            A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

            # Head weighting by image focus for this query distribution
            if head_weighting:
                # r @ A_h -> [T] for each head h; measure mass on image tokens
                v = torch.einsum('t,htt->ht', r, A)  # [H, T]
                focus = v[:, img_mask].mean(dim=1)  # [H]
                denom = (focus.sum() + 1e-8)
                if denom.item() == 0.0:
                    w = torch.ones_like(focus) / focus.numel()
                else:
                    w = (focus / denom).clamp(min=0.0)
                A = (A * w.view(-1, 1, 1)).sum(dim=0)  # [T, T]
            else:
                A = A.mean(dim=0)

            # Add identity and row-normalize (classic rollout)
            A = A + torch.eye(valid_len, device=device, dtype=A.dtype)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

            # r := r @ A
            r = r @ A

        # Keep image-token mass only, reshape to grid
        r_img = r[img_mask]
        grid_h, grid_w = _infer_patch_grid(model, r_img.shape[0])
        if grid_h * grid_w != r_img.shape[0]:
            return None

        # Sparse normalization for crispness
        if sparse_norm == "sparsemax":
            r_img = _sparsemax(r_img, dim=0)

        # Normalize to [0,1]
        rmin = float(r_img.min().item())
        rmax = float(r_img.max().item())
        if rmax > rmin:
            r_img = (r_img - rmin) / (rmax - rmin)
        return r_img.reshape(grid_h, grid_w).to(torch.float32)

    finally:
        # Restore attention impl
        if original_attn_impl is not None:
            try:
                if used_setter and attn_setter:
                    attn_setter(original_attn_impl)
                else:
                    model.config._attn_implementation = original_attn_impl
            except Exception:
                pass
