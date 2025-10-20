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
    cmd_lines: List[str] = []
    for ln in lines:
        m = pat.match(ln)
        if m:
            # Normalize function name casing to canonical (lowercase module, func as written)
            func = m.group(1)
            # Keep original content but ensure prefix is normalized to "pyautogui.<Func>"
            rest = ln.split(".", 1)[1] if "." in ln else ln
            cmd_lines.append(f"pyautogui.{rest}")
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

    def run(self, *, tokenizer, model, image_processor, device, conv_mode: str = "qwen_2", temperature: float = 0.2, top_p: Optional[float] = None, num_beams: int = 1, custom_prompt: Optional[str] = None, context_steps: int = 0) -> GuiBenchResult:
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

            if self.verbose:
                print(f"[DEBUG] output_ids shape: {output_ids.shape}")
                print(f"[DEBUG] input_ids shape: {input_ids.shape}")
                # output_ids should contain input+generated; compute actual generated count
                gen_count = output_ids.shape[1] - input_ids.shape[1] if output_ids.shape[1] >= input_ids.shape[1] else output_ids.shape[1]
                print(f"[DEBUG] Generated tokens: {gen_count}")

            pred_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # Save turn into plain-text history for the next step
            history.append((user_plain, pred_text))

            pred_cmds = _extract_pyautogui_lines(pred_text)
            exp_cmds = self._extract_expected_commands(step)

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

            # Score: count of expected commands matched by prediction
            step_hits = 0
            if exp_cmds:
                total_expected += len(exp_cmds)
                # Pre-compile tolerant regex for each expected command
                patterns = [_loose_regex_from_command(c) for c in exp_cmds]
                for pat in patterns:
                    if any(pat.match(p) for p in pred_cmds):
                        step_hits += 1
                matched += step_hits

        accuracy = (matched / total_expected) if total_expected > 0 else 0.0
        return GuiBenchResult(
            trajectory_dir=self.trajectory_dir,
            total_steps=len(steps),
            eval_steps=len(steps),
            expected_actions=total_expected,
            matched_actions=matched,
            accuracy=accuracy,
        )
