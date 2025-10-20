# Differential Encoding — A Mental Model

Think of the vision encoder as a painter rendering your desktop into a grid of tokens. Between frames, most of the canvas doesn’t change. Differential encoding teaches the painter to keep the old paint where it still matches the scene and only touch up the small regions that actually changed. Same story, fewer brush strokes.

This document explains, at a high level, how `predict.py` wires in the differential encoder and how `differential_vision/differential_encoder.py` implements the caching, change detection, and selective updates.

---

## Big Picture

- We process a stream of images (screenshots) with a VLM.
- For each new frame we reuse previously computed vision tokens wherever the pixels are unchanged.
- If only a few regions changed, we selectively recompute tokens just for those patches; otherwise we fall back to a full re-encode.
- We maintain robustness first (always correct), speed second (fast when changes are small).

---

## predict.py: The Wiring

`predict.py` is the CLI entry point that loads a FastVLM/LLaVA-style model and optionally enables differential vision encoding.

What it does:

1) Load model and pre/post-processing
   - Resolves `device`, loads tokenizer/model/image_processor via `load_pretrained_model`.
   - Builds the conversational prompt with the image token prefix (`DEFAULT_IM_*`).

2) Wrap `model.encode_images`
   - Creates `DifferentialVisionEncoder(model, ...)` unless `--disable-differential` is set.
   - Replaces `model.encode_images` with a thin shim that calls `diff_encoder.encode(...)`, but safely falls back to the original if anything goes wrong.
   - Restores the original `encode_images` at the end.

3) Drive sequences of frames
   - Accepts a single image, a list of images, a directory, or a root directory of sub-sequences.
   - For each sequence: calls `diff_encoder.reset_cache()` (so we don’t leak context across unrelated sequences).
   - For each frame: preprocess → `model.generate(..., images=...)` → decode tokens to text.
   - Optional `--benchmark` tallies per-frame timings and summarizes speedups; `--diff-print-stats` prints encoder stats.

4) Be dtype/device-friendly
   - Aligns tensors to the model’s dtype/device for stability and speed during generation.

Result: the rest of the system stays unaware; we just made the vision encoder “sticky” across frames.

---

## differential_encoder.py: The Core

The `DifferentialVisionEncoder` wraps a vision encoder (the “vision tower”) with a small cache and a simple patch-diff policy.

Key ideas:

- Cache: remember the last frame’s pixels and the resulting vision tokens.
- Patch grid: split the image into a grid of patches (e.g., 24×24 px for CLIP-like models). Either infer patch size from the tower config or derive from `image_size`/`grid_size`.
- Change detection: compute a per-patch absolute difference between consecutive frames and mark patches above a threshold as “changed”.
- Policy: 
  - 0 patches changed → return cached tokens.
  - Few patches changed → try a partial update (if supported); otherwise optionally skip (`--diff-skip-small`) or do a full re-encode.
  - Many patches changed → full re-encode.

### Fast Path vs Safe Path

- Fast path (partial update): Only attempted if the vision tower exposes `encode_patch(...)`. We batch all changed patches, run them through `encode_patch`, optionally through the model’s `mm_projector`, then scatter the new tokens back into the cached token grid. This avoids touching tokens for unchanged regions.
- Safe path (fallback): If partial updates aren’t supported, or too many patches changed, we re-encode the full image. The cache is then updated to the new tokens and pixels.

### Public API

```
features = encoder.encode(images, image_sizes=None, return_stats=False)
```

- Accepts a single image tensor `[1, C, H, W]` (batch>1 falls back to full encode).
- Returns the vision tokens (same shape/type as the underlying model produces).
- With `return_stats=True`, also returns a small dict summarizing which path was taken and how many patches changed.

### Internals at a Glance

1) Prepare input
   - Normalize input into a 4D tensor `[B, C, H, W]` with channels-first layout; move to the requested device.

2) Initialize cache as needed
   - On the very first frame or after resolution changes, perform a full encode and seed the cache with tokens and the raw RGB array.

3) Compute changed patches
   - Use `compute_patch_diff(prev, new, patch_size, threshold)` to get a boolean mask `[grid_h, grid_w]` of changed patches. This function is vectorized, handles edge patches fairly, and normalizes differences by valid pixel counts.

4) Decide what to do
   - 0 changed → return cached tokens.
   - ≤ `max_changed_patches` and `encode_patch` exists → partial update.
   - ≤ `max_changed_patches` but no `encode_patch` → if `skip_small_updates` is set, reuse the cache and skip; otherwise full re-encode.
   - > `max_changed_patches` → full re-encode.

5) Partial update details
   - Collect all changed `(i, j)` patch coordinates.
   - Extract each patch (optionally with a small context window via `extract_patch_window`) from the NEW image.
   - Batch-encode patches once via `vision_tower.encode_patch(...)` (passing positions if supported).
   - If a multimodal projector exists (`mm_projector`), apply it in a single batched call.
   - Scatter the resulting per-patch tokens back into the flat token grid at their positions; return the updated cache.

6) Stats
   - Tracks counts of full/partial/skipped/cache-hit frames, total frames, and aggregate changed patches to compute an average change ratio. Useful for benchmarking.

---

## Why This Works Well on Screens

Screens are mostly static. The mouse moves, a button flashes, text blinks—small, local edits on a large canvas. By aligning the encoder’s compute with the spatial sparsity of changes, we reduce redundant work dramatically without changing the model’s language or reasoning stack.

In practice you get:

- Stable first-frame latency (full encode), then much faster subsequent frames.
- Significant speedups when interactions are local (cursor, clicks, small UI updates).
- Automatic fallback when the scene changes a lot (scrolling, window switches).

---

## Tuning Knobs

- `--diff-threshold` (`diff_threshold`): Sensitivity of patch change detection. Lower → more patches marked as changed (safer, more compute). Higher → fewer patches (faster, risk of stale tokens if too high).
- `--diff-max-changed-patches` (`max_changed_patches`): Upper bound for partial updates. Beyond this, we full re-encode.
- `--diff-patch-size` (`patch_size`): Override the patch size if not inferred; must match the vision tower’s tokenization.
- `--diff-skip-small` (`skip_small_updates`): If only a few patches changed and partial updates aren’t supported, reuse cache anyway.

Defaults are conservative and prioritize correctness over maximal speed.

---

## Failure Modes and Safeguards

- No `encode_patch`: We still benefit from full-image caching (return cached tokens when 0 patches changed) and skip-small mode; otherwise we fall back to full encodes safely.
- Resolution changes: Any change of `[H, W]` triggers a full re-encode and cache reset.
- Batches: Batch size > 1 triggers a full encode for simplicity and correctness.

These constraints keep the integration robust with standard CLIP/LLaVA stacks.

---

## Pseudocode (Mental Model)

```python
state = { tokens: None, pixels: None }

def encode(frame):
    if state.tokens is None or frame.shape != state.pixels.shape:
        state.tokens = full_encode(frame)
        state.pixels = frame
        return state.tokens

    changed = patch_diff(state.pixels, frame, threshold)
    k = count(changed)

    if k == 0:
        state.pixels = frame
        return state.tokens

    if k <= max_changed and supports_encode_patch:
        updates = encode_patches(frame, where=changed)
        state.tokens = scatter(state.tokens, updates, where=changed)
        state.pixels = frame
        return state.tokens

    if k <= max_changed and skip_small_updates:
        state.pixels = frame
        return state.tokens

    state.tokens = full_encode(frame)
    state.pixels = frame
    return state.tokens
```

---

## Where to Look in Code

- predict.py: sequence handling, model loading, and the `encode_images` hook that delegates to the differential encoder; begins at `predict(args)`.
- differential_vision/differential_encoder.py: `DifferentialVisionEncoder` implementation (cache, patch diff, partial update, stats).
- differential_vision/patch_utils.py: vectorized patch diff (`compute_patch_diff`), patch extraction, and simple visualizations.

---

## Takeaway

We embrace the temporal sparsity of screens. The model still “sees” the whole image, but we avoid re-painting what hasn’t changed. It’s simple, robust, and yields big wins on real human-computer interaction streams.

