#!/usr/bin/env python3
import os
import sys
import math
from PIL import Image
from collections import Counter, defaultdict


PINPOINTS = [
    (1024, 1024),
    (2048, 1024),
    (1024, 2048),
    (2048, 2048),
]


def select_best_resolution(original_size, possible_resolutions):
    ow, oh = original_size
    best = None
    max_effective = -1
    min_wasted = 1 << 60
    for (W, H) in possible_resolutions:
        scale = min(W / ow, H / oh)
        dw, dh = int(ow * scale), int(oh * scale)
        effective = min(dw * dh, ow * oh)
        wasted = (W * H) - effective
        if effective > max_effective or (effective == max_effective and wasted < min_wasted):
            max_effective = effective
            min_wasted = wasted
            best = (W, H)
    return best


def sample_images(root, limit=50):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    picked = []
    for base, _, files in os.walk(root):
        for f in files:
            if len(picked) >= limit:
                return picked
            ext = os.path.splitext(f)[1].lower()
            if ext in exts:
                picked.append(os.path.join(base, f))
        # Avoid deep recursion explosion by sampling early across many folders
        if len(picked) >= limit:
            break
    return picked


def main():
    if len(sys.argv) < 2:
        print("Usage: audit_anyres.py <images_dir> [<images_dir> ...]")
        sys.exit(1)

    tile = int(os.environ.get("ANYRES_TILE", "1024"))
    print({"tile": tile, "pinpoints": PINPOINTS})

    totals = 0
    orientations = Counter()
    best_counts = Counter()
    grid_counts = Counter()
    examples = defaultdict(list)

    for d in sys.argv[1:]:
        imgs = sample_images(d, limit=80)
        for p in imgs:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                continue
            totals += 1
            orientations["portrait" if h > w else "landscape"] += 1
            best = select_best_resolution((w, h), PINPOINTS)
            best_counts[best] += 1
            grid = (best[0] // tile, best[1] // tile)
            grid_counts[grid] += 1
            if len(examples[best]) < 3:
                examples[best].append((p, (w, h)))

    print("totals", totals)
    print("orientations", dict(orientations))
    print("best_counts", {str(k): v for k, v in best_counts.items()})
    print("grid_counts", {str(k): v for k, v in grid_counts.items()})
    for k, v in examples.items():
        print("example", k, v)


if __name__ == "__main__":
    main()

